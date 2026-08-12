"""eval_acceptance_tree.py — TREE speculative-decoding σ + wall-time speedup.

Does NOT touch scripts/eval_acceptance.py (the validated chain σ=1.95). Shares
loaders / vocab mask / vanilla_greedy via decode.common, and tree primitives via
decode.tree.

Modes (--mode):
  tree        : run tree decoding, report σ_tree + mean tree width + tok/s.
  regression  : BLOCKING gate. fanout=[1,1,1] tree must reproduce the chain's
                emitted tokens byte-for-byte on N image prompts (σ back to ~1.95).
                Compares against scripts.eval_acceptance.run_one_prompt directly.

Per-round structure (design doc §6, with the M-RoPE fix):
  heads(h_t) -> per-level logits -> build_tree -> mask+positions(cont_base)
   -> verify forward (1, custom 4D mask) -> accept -> emit accepted+bonus
   -> reorg KV (safe recompute by default) -> bonus forward -> next h_t / base_pred
  cont_base = continuation_base(base, P) = P + rope_delta  (MANDATORY on images).
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/scratch/li96/mz9869/tmp_hf_download")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decode.common import (  # noqa: E402
    ARCHIVE_GATE_FAIL,
    ARCHIVE_GATE_INCOMPLETE,
    ARCHIVE_GATE_NOT_RUN,
    ARCHIVE_GATE_PASS,
    O0_CLAIM_OFFICIAL,
    O0_KERNEL_BAND,
    apply_max_pixels,
    archive_gate_exit_code,
    archive_gate_status,
    argmax_masked,
    classify_o0_vs_ref,
    compare_token_sequences,
    continuation_base,
    cfg_attr,
    filter_prompts,
    greedy_byte_exact_pass,
    greedy_candidate_probe_at,
    greedy_numerical_safety_pass,
    load_base,
    load_head,
    load_o0_archive,
    make_image_inputs,
    o0_archive_triggers_fail,
    sha256_file,
    truncate_emit_path,
    validate_o0_archive_not_self,
    vanilla_greedy,
)
from decode.tree import (  # noqa: E402
    accept,
    build_mask_and_positions,
    build_tree,
    per_depth_widths,
    reorg_kv_gather,
    reorg_kv_safe,
    tree_tokens,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/scratch/li96/mz9869/medusa_outputs/linked_medusa_v1_full/ckpt_best.pt")
    p.add_argument("--manifest", default="/g/data/li96/mz9869/data/llava_subset_2k.json")
    p.add_argument("--images-dir", default="/g/data/li96/mz9869/data/coco_subset")
    p.add_argument("--out", default="/scratch/li96/mz9869/medusa_outputs/linked_medusa_v1_full/eval_acceptance_tree_v1_fp16.json")
    p.add_argument("--mode",
                   choices=["tree", "regression", "validate_reorg", "foldbonus_ab",
                            "greedy_agreement", "greedy_e2e"],
                   default="tree")
    p.add_argument("--fold-bonus", action="store_true",
                   help="(3c) in tree mode, decode with the bonus forward folded into the "
                        "next round's verify (run_one_prompt_tree_folded)")
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--min-ref-words", type=int, default=80)
    p.add_argument("--ordered-manifest", action="store_true",
                   help="Keep manifest file order (no Random shuffle). Required for "
                        "nested-300 / MM-Vet fixed manifests so prompt ids stay aligned.")
    p.add_argument("--check-greedy-bytes", action="store_true",
                   help="O0 dual-verdict (2026-08 Tier D): (1) archive reproducibility "
                        "four-state NOT_RUN/INCOMPLETE/PASS/FAIL (byte-exact vs archive); "
                        "(2) vs independent greedy: greedy_byte_exact_pass="
                        "(n_exact==n_prompts) expected FALSE; "
                        "greedy_numerical_safety_pass=(no len_boundary/hard). "
                        "Claim: every first mid-sequence divergence is a "
                        "candidate-specific near_tie "
                        "(0 <= logit[greedy_top1]-logit[spec_tok] <= 0.15). "
                        "Exit 2=archive FAIL, 3=safety FAIL, 4=both; "
                        "5=NOT_RUN, 6=INCOMPLETE when --o0-archive set.")
    p.add_argument("--o0-archive", default=None,
                   help="Frozen speculative archive for archive gate. Requires "
                        "--check-greedy-bytes. Zero coverage → NOT_RUN; partial → "
                        "INCOMPLETE; full → PASS/FAIL.")
    p.add_argument("--o0-write-archive", default=None,
                   help="Write speculative archive (all prompts) + sha when archive gate "
                        "is PASS or NOT_RUN (fresh write). Sidecar runner=v2_hardcap.")
    p.add_argument("--o0-write-greedy-archive", default=None,
                   help="On numerical_safety PASS, write archive of byte-exact "
                        "(kind=match) prompts only. Sidecar runner=v2_hardcap.")
    p.add_argument(
        "--o0-cli-selftest",
        choices=["archive_pass", "archive_fail", "archive_not_run",
                 "archive_incomplete", "safety_fail"],
        default=None,
        help="Test-only: skip model load; exercise O0 exit-code path via subprocess.",
    )
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--fanout", type=int, nargs="+", default=[4, 3, 2])
    p.add_argument("--max-nodes", type=int, default=16)
    p.add_argument("--reorg", choices=["safe", "gather"], default="gather",
                   help="gather=index_select on cache (DEFAULT: proven byte-identical to the "
                        "validated chain, faster); safe=recompute accepted path (fp16-drifts vs "
                        "chain via is_causal kernel). See docs/tree_decoding_design.md §0.2.")
    p.add_argument("--no-depth1-floor", action="store_true")
    p.add_argument("--tree-builder", choices=["static", "dynamic"], default="static",
                   help="folded tree constructor: static=build_tree_folded (DEFAULT); "
                        "dynamic=build_tree_folded_dynamic (best-first + post-hoc floor). "
                        "Non-folded path ignores this flag.")
    p.add_argument("--measure-greedy", action="store_true",
                   help="also run vanilla greedy on the same prompts for tok/s baseline")
    p.add_argument("--e2e-wall", action="store_true",
                   help="Official speed_300 protocol (2026-08): per-prompt end-to-end "
                        "CUDA-synced wall clock covering the full generate loop "
                        "(build+verify+accept+KV reorg). Warmup 1 prompt first. "
                        "Paired speedup = mean(greedy_wall/spec_wall) ± SE. "
                        "Segmented/profile timers are diagnostic only — not official.")
    p.add_argument("--measure-chain", action="store_true",
                   help="also run the linear-chain decoder (eval_acceptance.run_one_prompt) "
                        "for its tok/s — isolates whether the 2.1B head overhead alone "
                        "already makes speculative decoding net-negative vs greedy")
    p.add_argument("--profile", action="store_true",
                   help="(3a) CUDA-synced per-round wall-time breakdown: "
                        "head_fwd / verify_fwd / reorg / bonus_fwd / other. "
                        "Use with --measure-greedy to express bonus_fwd in base-units.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--max-pixels", type=int, default=None,
                   help="Cap processor image_processor.max_pixels (A-batch: 501760). "
                        "None = model/processor default.")
    args = p.parse_args()
    if args.mode == "regression":
        args.fanout = [1, 1, 1]
        args.max_nodes = max(args.max_nodes, len(args.fanout))
        if args.n_prompts > 20:
            args.n_prompts = 5
    return args


def _sync_now(profile: bool) -> float:
    """CUDA-synced timestamp (only syncs when profiling, to avoid perturbing prod)."""
    if profile:
        torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def run_one_prompt_tree(
    base, head, processor, prompt, images_dir, max_new, eos_id,
    fanout, max_nodes, depth1_floor, reorg, device="cuda:0", profile=False,
) -> Dict:
    dtype = base.dtype
    inputs = make_image_inputs(processor, prompt["question"], images_dir / prompt["image"], device)

    # prefill
    out = base(**inputs, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    P = past_kv.get_seq_length()
    h_t = out.hidden_states[-1][0, -1, :].clone()
    base_pred_root = argmax_masked(out.logits[0, -1, :])   # round-0 init (carried like chain)
    del out
    torch.cuda.empty_cache()

    emitted: List[int] = []
    accept_log: List[int] = []
    width_log: List[int] = []
    depth_width_log: List[List[int]] = []
    depth_accept_counter: Counter = Counter()   # how many accepted tokens landed at each depth
    rounds = 0
    hit_eos = False
    # per-component wall-time (seconds), accumulated over rounds (3a)
    t_head = t_verify = t_reorg = t_bonus = t_other = 0.0

    while len(emitted) < max_new:
        remaining = max_new - len(emitted)
        rounds += 1
        r0 = _sync_now(profile)
        cont_base = continuation_base(base, P)

        # heads draft
        all_logits = head(h_t.view(1, 1, -1).half())
        nodes = build_tree(all_logits, fanout, max_nodes, depth1_floor=depth1_floor)
        del all_logits
        N = len(nodes)
        width_log.append(N)
        depth_width_log.append(per_depth_widths(nodes, head.num_heads))
        mask, pos = build_mask_and_positions(nodes, P, cont_base, dtype, device)
        toks = tree_tokens(nodes, device)
        t1 = _sync_now(profile); t_head += t1 - r0

        v_out = base(input_ids=toks, attention_mask=mask, past_key_values=past_kv,
                     position_ids=pos, use_cache=True, output_hidden_states=False)
        v_logits = v_out.logits[0]   # (N, V)
        t2 = _sync_now(profile); t_verify += t2 - t1

        accepted, accept_len, bonus, acc_depths = accept(nodes, v_logits, base_pred_root)
        accept_log.append(accept_len)
        for d in acc_depths:
            depth_accept_counter[d] += 1

        accepted_tokens = [nodes[i].token for i in accepted]
        # Hard cap: only emit up to `remaining`; stop at EOS. (runner v2)
        path = accepted_tokens + [bonus]
        to_emit, hit_eos = truncate_emit_path(path, remaining, eos_id)
        emitted.extend(to_emit)
        # No consumer for further KV work once capped or EOS'd.
        if hit_eos or len(emitted) >= max_new:
            del v_out
            break
        # Continuing requires the full accepted path + bonus were emitted.
        if len(to_emit) < len(path):
            del v_out
            break

        # reorg KV: keep prefix + accepted path
        ta = _sync_now(profile)
        if reorg == "gather":
            reorg_kv_gather(v_out.past_key_values, P, accepted, device)
            past_kv = v_out.past_key_values
        else:  # safe
            past_kv = v_out.past_key_values
            reorg_kv_safe(base, past_kv, P, accepted_tokens, cont_base, device)
        del v_out
        tb = _sync_now(profile); t_reorg += tb - ta

        # bonus forward -> next h_t / base_pred_root, extends cache by 1
        P = past_kv.get_seq_length()
        cont_base_bonus = continuation_base(base, P)
        b_out = base(input_ids=torch.tensor([[bonus]], device=device),
                     past_key_values=past_kv,
                     position_ids=torch.tensor([[cont_base_bonus]], device=device),
                     use_cache=True, output_hidden_states=True)
        past_kv = b_out.past_key_values
        h_t = b_out.hidden_states[-1][0, -1, :].clone()
        base_pred_root = argmax_masked(b_out.logits[0, -1, :])
        P = past_kv.get_seq_length()
        del b_out
        tc = _sync_now(profile); t_bonus += tc - tb
        t_other += tc - r0 - (t1 - r0) - (t2 - t1) - (tb - ta) - (tc - tb)

    total = len(emitted)
    assert total <= max_new, f"v2 hardcap violated: emit={total} > max_new={max_new}"
    sigma = total / max(1, rounds)
    res = {
        "id": prompt["id"], "image": prompt["image"],
        "answer_word_count": prompt["answer_word_count"],
        "sigma": sigma, "rounds": rounds, "total_emitted": total, "hit_eos": hit_eos,
        "accept_log": accept_log, "mean_width": statistics.mean(width_log) if width_log else 0,
        "mean_depth_widths": [statistics.mean(c) for c in zip(*depth_width_log)] if depth_width_log else [],
        "depth_accept_counts": dict(depth_accept_counter),
        "accept_distribution": dict(Counter(accept_log)),
        "emitted_tokens": emitted,
        "runner": "v2_hardcap",
    }
    if profile:
        res["profile_s"] = {"head": t_head, "verify": t_verify, "reorg": t_reorg,
                            "bonus": t_bonus, "other": t_other}
    return res


@torch.no_grad()
def run_one_prompt_tree_folded(
    base, head, processor, prompt, images_dir, max_new, eos_id,
    fanout, max_nodes, device="cuda:0", profile=False, depth1_floor=True,
    skip_head0_lm_head=True,
    tree_builder: str = "static",
) -> Dict:
    """(3c) Fold the per-round bonus forward away.

    Draft each round from the LAST-ACCEPTED node's hidden (verify runs with
    output_hidden_states=True); the known bonus is the FORCED depth-1 root of the
    next tree (build_tree_folded), so its KV is produced by the verify forward
    itself — no separate bonus forward. Reorg is always gather. Costs one fewer
    speculative draft layer (head_0 re-predicts the known bonus). See §0.4.

    skip_head0_lm_head (default True): head_0's logits have no consumer in the
    folded tree (the depth-1 root is forced to the known bonus), so its lm_head
    GEMV (~1.09GB weight read/round) is skipped; h_0' still feeds the chain, so
    all tree-building logits are bit-identical and sigma must not change. The
    head0_top1 observation is unavailable in this mode (reported as None);
    _greedy_agreement passes False to keep that diagnostic.
    """
    from decode.tree import build_tree_folded, build_tree_folded_dynamic
    _builders = {"static": build_tree_folded, "dynamic": build_tree_folded_dynamic}
    if tree_builder not in _builders:
        raise ValueError(f"unknown tree_builder={tree_builder!r}")
    _build = _builders[tree_builder]
    dtype = base.dtype
    inputs = make_image_inputs(processor, prompt["question"], images_dir / prompt["image"], device)
    out = base(**inputs, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    P = past_kv.get_seq_length()
    h_anchor = out.hidden_states[-1][0, -1, :].clone()
    known_next = argmax_masked(out.logits[0, -1, :])   # base's confirmed next token (the root)
    del out
    torch.cuda.empty_cache()

    emitted: List[int] = []
    accept_log: List[int] = []
    width_log: List[int] = []
    depth_width_log: List[List[int]] = []
    depth_accept_counter: Counter = Counter()
    rounds = 0
    hit_eos = False
    head0_hits = 0   # OBSERVATION ONLY (not a gate): head_0 is unused in build_tree_folded
                     # (root is forced); this predicts the bonus itself from the last-accepted
                     # hidden — a different task than baseline head_0, so a low value is expected.
    head0_by_depth: Counter = Counter()
    rounds_by_depth: Counter = Counter()   # rounds bucketed by anchor source depth
    K = head.num_heads
    acc_gt_by_depth = {k: Counter() for k in range(K)}   # real bug detector: rounds with
                     # accept_len>k bucketed by anchor depth (head_1=k1, head_2=k2)
    anchor_depth = 0   # 0=prefill anchor; else depth of last-accepted node feeding h_anchor
    t_head = t_verify = t_reorg = t_other = 0.0

    # Truncate the head chain to the tree's max speculative depth: trailing
    # zero-fanout entries contribute no nodes, so running those heads is pure
    # per-round waste (2 unused lm_head GEMVs cost ~10% round time on V100 for
    # a 5-head module on a depth-2 tree). The linked chain is sequential
    # (head_k depends only on heads 0..k-1), so truncation is exact.
    n_active = len(fanout)
    while n_active > 1 and fanout[n_active - 1] == 0:
        n_active -= 1
    fanout_active = list(fanout[:n_active])
    # Dynamic builder takes exactly K-1 speculative widths (depths 2..K).
    # CLI --fanout keeps a unused head-0 slot (forced root); strip here.
    if tree_builder == "dynamic":
        if len(fanout_active) < 2:
            raise ValueError(f"dynamic builder needs ≥1 speculative width; fanout={fanout_active}")
        build_widths = fanout_active[1:]
    else:
        build_widths = fanout_active

    while len(emitted) < max_new:
        remaining = max_new - len(emitted)
        rounds += 1
        r0 = _sync_now(profile)
        cont_base = continuation_base(base, P)
        with torch.no_grad():
            cond_embed = base.get_input_embeddings()(
                torch.tensor([[known_next]], device=device, dtype=torch.long)
            )
        all_logits = head(h_anchor.view(1, 1, -1).half(), max_heads=n_active,
                          skip_head0_lm_head=skip_head0_lm_head,
                          cond_embed=cond_embed)
        rounds_by_depth[anchor_depth] += 1
        if all_logits[0] is not None and argmax_masked(all_logits[0].reshape(-1)) == known_next:
            head0_hits += 1
            head0_by_depth[anchor_depth] += 1
        nodes = _build(all_logits, known_next, build_widths, max_nodes,
                       depth1_floor=depth1_floor)
        if tree_builder == "dynamic" and depth1_floor and build_widths:
            # End-to-end: CLI speculative width[0] must realize as depth-2 width.
            d2 = sum(1 for nd in nodes if nd.depth == 2)
            if d2 != build_widths[0]:
                raise RuntimeError(
                    f"CLI→depth-2 width mismatch: realized {d2} != cand_k[0]={build_widths[0]} "
                    f"(fanout={fanout_active})"
                )
        del all_logits
        width_log.append(len(nodes))
        depth_width_log.append(per_depth_widths(nodes, head.num_heads))
        mask, pos = build_mask_and_positions(nodes, P, cont_base, dtype, device)
        toks = tree_tokens(nodes, device)
        t1 = _sync_now(profile); t_head += t1 - r0

        v_out = base(input_ids=toks, attention_mask=mask, past_key_values=past_kv,
                     position_ids=pos, use_cache=True, output_hidden_states=True)
        v_logits = v_out.logits[0]
        v_hidden = v_out.hidden_states[-1][0]   # (N, H)
        t2 = _sync_now(profile); t_verify += t2 - t1

        # root (== known_next) is base's confirmed token at the anchor -> always accepted
        accepted, accept_len, next_known, acc_depths = accept(nodes, v_logits, known_next)
        accept_log.append(accept_len)
        for k in range(K):
            if accept_len > k:
                acc_gt_by_depth[k][anchor_depth] += 1
        for d in acc_depths:
            depth_accept_counter[d] += 1
        accepted_tokens = [nodes[i].token for i in accepted]
        to_emit, hit_eos = truncate_emit_path(accepted_tokens, remaining, eos_id)
        emitted.extend(to_emit)
        # Cap / EOS: no consumer for reorg or trailing work this round.
        if hit_eos or len(emitted) >= max_new:
            del v_out
            break
        if len(to_emit) < len(accepted_tokens):
            del v_out
            break

        # carry: new anchor hidden + new known bonus from the last accepted node
        last = accepted[-1]
        h_anchor = v_hidden[last].clone()
        known_next = next_known
        anchor_depth = acc_depths[-1]   # depth of node feeding next round's h_anchor

        ta = _sync_now(profile)
        reorg_kv_gather(v_out.past_key_values, P, accepted, device)
        past_kv = v_out.past_key_values
        del v_out
        P = past_kv.get_seq_length()
        tb = _sync_now(profile); t_reorg += tb - ta
        t_other += tb - r0 - (t1 - r0) - (t2 - t1) - (tb - ta)

    # emit the trailing confirmed bonus (next round's would-be root) if room & not EOS
    if not hit_eos and len(emitted) < max_new:
        bonus_emit, bonus_eos = truncate_emit_path([known_next], max_new - len(emitted), eos_id)
        emitted.extend(bonus_emit)
        hit_eos = hit_eos or bonus_eos

    total = len(emitted)
    assert total <= max_new, f"v2 hardcap violated: emit={total} > max_new={max_new}"
    sigma = total / max(1, rounds)
    K = head.num_heads
    # per-position accept: folded[0]=1.0 (root/bonus, always), [1]=head_1, [2]=head_2
    ppa = [sum(int(a > k) for a in accept_log) / max(1, rounds) for k in range(K)]
    res = {
        "id": prompt["id"], "image": prompt["image"],
        "answer_word_count": prompt["answer_word_count"],
        "sigma": sigma, "rounds": rounds, "total_emitted": total, "hit_eos": hit_eos,
        "accept_log": accept_log, "mean_width": statistics.mean(width_log) if width_log else 0,
        "mean_depth_widths": [statistics.mean(c) for c in zip(*depth_width_log)] if depth_width_log else [],
        "depth_accept_counts": dict(depth_accept_counter),
        "accept_distribution": dict(Counter(accept_log)),
        "per_position_accept": ppa,
        # plumbing health; None when head_0's lm_head is skipped (no consumer)
        "head0_top1_rate": (None if skip_head0_lm_head
                            else head0_hits / max(1, rounds)),
        "head0_rate_by_anchor_depth": (None if skip_head0_lm_head else
                                       {int(d): head0_by_depth[d] / rounds_by_depth[d]
                                        for d in sorted(rounds_by_depth)}),
        "rounds_by_anchor_depth": {int(d): rounds_by_depth[d] for d in sorted(rounds_by_depth)},
        "headk_accept_by_anchor_depth": {   # head_k accept rate by anchor depth (k>=1 = real detector)
            int(k): {int(d): acc_gt_by_depth[k][d] / rounds_by_depth[d]
                     for d in sorted(rounds_by_depth)}
            for k in range(K)},
        "emitted_tokens": emitted,
        "runner": "v2_hardcap",
    }
    if profile:
        res["profile_s"] = {"head": t_head, "verify": t_verify, "reorg": t_reorg,
                            "bonus": 0.0, "other": t_other}
    return res


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _e2e_wall_call(fn):
    """CUDA-synced end-to-end wall for one callable (returns (result, wall_s))."""
    _cuda_sync()
    t0 = time.perf_counter()
    out = fn()
    _cuda_sync()
    return out, time.perf_counter() - t0


def token_seq_sha256(tokens) -> Optional[str]:
    """SHA-256 over a token-id sequence (comma-joined decimal ids, utf-8).

    Lets a reviewer bind a divergence row to the exact greedy prefix that
    produced it, instead of the ±5-token context window.
    """
    if tokens is None:
        return None
    return hashlib.sha256(
        ",".join(str(int(t)) for t in tokens).encode("utf-8")
    ).hexdigest()


def o0_run_fingerprint(args: argparse.Namespace) -> Dict:
    """Prompt / config / model identity for one O0 report.

    Written from this version on; **not** backfilled into older reports.
    """
    def sha_file(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        p = Path(path)
        if not p.is_file():
            return None
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    return {
        "schema": "o0_report_fingerprint/v1",
        "prompt_set": {
            "manifest": args.manifest,
            "manifest_sha256": sha_file(args.manifest),
            "n_prompts": args.n_prompts,
            "min_ref_words": args.min_ref_words,
            "ordered_manifest": bool(args.ordered_manifest),
            "seed": args.seed,
            "images_dir": args.images_dir,
        },
        "config": {
            "mode": args.mode,
            "fanout": list(args.fanout),
            "max_nodes": args.max_nodes,
            "max_new_tokens": args.max_new_tokens,
            "tree_builder": args.tree_builder,
            "reorg": args.reorg,
            "fold_bonus": bool(args.fold_bonus),
            "max_pixels": args.max_pixels,
            "kernel_band": O0_KERNEL_BAND,
        },
        "model": {
            "model_id": args.model_id,
            "ckpt": args.ckpt,
            "ckpt_sha256": sha_file(args.ckpt),
            "torch": torch.__version__,
            "dtype": str(getattr(torch, "float16", None)),
        },
    }


def _combine_o0_exit(
    arch_status: str,
    archive_provided: bool,
    safety_fail: bool,
) -> int:
    """Same exit composition as the production path at end of main()."""
    arch_exit = archive_gate_exit_code(arch_status, archive_provided)
    if arch_exit is not None and safety_fail and arch_exit == 2:
        return 4
    if arch_exit is not None:
        return arch_exit
    if safety_fail:
        return 3
    return 0


def _run_o0_cli_selftest(args: argparse.Namespace) -> int:
    """No-GPU CLI selftest for archive/safety exit codes."""
    scenario = args.o0_cli_selftest
    n = 4
    archive_provided = True
    if scenario == "archive_pass":
        status, covered, fails, n_hard = ARCHIVE_GATE_PASS, n, 0, 0
    elif scenario == "archive_fail":
        status, covered, fails, n_hard = ARCHIVE_GATE_FAIL, n, 1, 0
    elif scenario == "archive_not_run":
        status, covered, fails, n_hard = ARCHIVE_GATE_NOT_RUN, 0, 0, 0
    elif scenario == "archive_incomplete":
        status, covered, fails, n_hard = ARCHIVE_GATE_INCOMPLETE, 2, 0, 0
    else:  # safety_fail — archive not provided / informational NOT_RUN
        archive_provided = False
        status = archive_gate_status(False, 0, n, 0)
        covered, fails, n_hard = 0, 0, 1
    # Consistency check against helpers
    assert status == archive_gate_status(archive_provided, covered, n, fails)
    safety_fail = not greedy_numerical_safety_pass(0, n_hard)
    code = _combine_o0_exit(status, archive_provided, safety_fail)
    print(f"[o0-cli-selftest] scenario={scenario} archive_gate={status} "
          f"safety_fail={safety_fail} exit={code}")
    return code


def main() -> int:
    args = parse_args()
    # Every O0 flag is inert without the O0 pass; accepting one silently would
    # produce a run that looks archived/gated but is neither.
    for flag, value in (
        ("--o0-archive", args.o0_archive),
        ("--o0-write-archive", args.o0_write_archive),
        ("--o0-write-greedy-archive", args.o0_write_greedy_archive),
    ):
        if value and not args.check_greedy_bytes:
            raise SystemExit(
                f"error: {flag} requires --check-greedy-bytes "
                "(refusing to silently ignore it)"
            )
    if args.o0_cli_selftest:
        if not args.check_greedy_bytes:
            raise SystemExit(
                "error: --o0-cli-selftest requires --check-greedy-bytes"
            )
        return _run_o0_cli_selftest(args)
    for k, v in vars(args).items():
        print(f"[args] {k} = {v}")

    base, processor = load_base(args.model_id)
    if args.max_pixels is not None:
        apply_max_pixels(processor, args.max_pixels)
        print(f"[cap] max_pixels={args.max_pixels}  "
              f"ip.size={getattr(processor.image_processor, 'size', None)}")
    torch.cuda.reset_peak_memory_stats()
    head = load_head(args.ckpt, cfg_attr(base.config, "hidden_size"),
                     cfg_attr(base.config, "vocab_size"))
    eos_id = processor.tokenizer.eos_token_id
    assert len(args.fanout) == head.num_heads, \
        f"fanout {args.fanout} must have len == num_heads {head.num_heads}"
    images_dir = Path(args.images_dir)
    prompts = filter_prompts(
        args.manifest, args.min_ref_words, args.seed, ordered=args.ordered_manifest,
    )[: args.n_prompts]
    print(f"[data] {len(prompts)} prompts (>= {args.min_ref_words} ref words; "
          f"ordered={args.ordered_manifest})")

    if args.mode == "regression":
        return _regression(base, head, processor, prompts, images_dir, eos_id, args)
    if args.mode == "validate_reorg":
        return _validate_reorg(base, head, processor, prompts, images_dir, eos_id, args)
    if args.mode == "foldbonus_ab":
        return _foldbonus_ab(base, head, processor, prompts, images_dir, eos_id, args)
    if args.mode == "greedy_agreement":
        return _greedy_agreement(base, head, processor, prompts, images_dir, eos_id, args)
    if args.mode == "greedy_e2e":
        return _greedy_e2e(base, processor, prompts, images_dir, eos_id, args)

    def _run_tree(p):
        if args.fold_bonus:
            return run_one_prompt_tree_folded(
                base, head, processor, p, images_dir, args.max_new_tokens,
                eos_id, args.fanout, args.max_nodes, profile=args.profile,
                depth1_floor=not args.no_depth1_floor,
                tree_builder=args.tree_builder,
            )
        return run_one_prompt_tree(
            base, head, processor, p, images_dir, args.max_new_tokens,
            eos_id, args.fanout, args.max_nodes,
            not args.no_depth1_floor, args.reorg, profile=args.profile,
        )

    # ---- tree mode ----
    if args.e2e_wall and prompts:
        print("[e2e-wall] warmup prompt 0 (timing discarded)…")
        _run_tree(prompts[0])
        _cuda_sync()

    results = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        if args.e2e_wall:
            r, wall = _e2e_wall_call(lambda p=p: _run_tree(p))
            r["e2e_wall_s"] = wall
            dt = wall
        else:
            rt = time.time()
            r = _run_tree(p)
            dt = time.time() - rt
        print(f"[tree {i+1:>3}/{len(prompts)}] σ={r['sigma']:.3f} emit={r['total_emitted']} "
              f"rounds={r['rounds']} N̄={r['mean_width']:.1f} dist={r['accept_distribution']} "
              f"eos={r['hit_eos']} ({dt:.1f}s)")
        results.append(r)
    decode_wall = time.time() - t0

    greedy_tok_s = None
    greedy_e2e_walls: List[float] = []
    # Dual-verdict O0: archive reproducibility + greedy numerical safety
    o0_spec_fails: List[Dict] = []
    o0_greedy_safety_fails: List[Dict] = []  # hard + len_boundary
    o0_near_ties: List[Dict] = []
    o0_exact_ids: List[str] = []
    o0_per_prompt: List[Dict] = []
    o0_archive_map: Dict[str, List[int]] = {}
    o0_archive_meta: Optional[Dict] = None
    archive_provided = bool(args.check_greedy_bytes and args.o0_archive)
    if archive_provided:
        o0_archive_meta = validate_o0_archive_not_self(
            args.o0_archive,
            args.out,
            run_job_id=os.environ.get("PBS_JOBID"),
        )
        o0_archive_map = load_o0_archive(args.o0_archive)
        print(f"[O0] archive loaded: {len(o0_archive_map)} prompts from {args.o0_archive}")
        print(f"[O0] archive provenance: sha256={o0_archive_meta['sha256'][:16]}… "
              f"job_id={o0_archive_meta.get('job_id')} "
              f"anti-self OK (path≠out)")

    # Greedy sequences (timing MUST exclude O0 gap probes — those are extra
    # forwards and would collapse reported greedy tok/s ~30 → ~21).
    # --e2e-wall alone times the tree only; pass --measure-greedy to also time
    # greedy under the same sync protocol (or use --mode greedy_e2e once + join).
    greedy_seqs: List[List[int]] = []
    need_greedy = args.measure_greedy or args.check_greedy_bytes
    if need_greedy:
        if args.e2e_wall and prompts:
            print("[e2e-wall] greedy warmup prompt 0 (timing discarded)…")
            vanilla_greedy(base, processor, prompts[0]["question"],
                           images_dir / prompts[0]["image"], args.max_new_tokens, eos_id)
            _cuda_sync()
        gt0 = time.time(); gtok = 0
        for i, p in enumerate(prompts):
            if args.e2e_wall:
                g, gwall = _e2e_wall_call(
                    lambda p=p: vanilla_greedy(
                        base, processor, p["question"], images_dir / p["image"],
                        args.max_new_tokens, eos_id,
                    )
                )
                greedy_e2e_walls.append(gwall)
                results[i]["greedy_e2e_wall_s"] = gwall
                results[i]["paired_speedup"] = (
                    gwall / results[i]["e2e_wall_s"] if results[i]["e2e_wall_s"] > 0 else None
                )
            else:
                g = vanilla_greedy(base, processor, p["question"], images_dir / p["image"],
                                   args.max_new_tokens, eos_id)
            greedy_seqs.append(g)
            gtok += len(g)
        greedy_tok_s = gtok / (time.time() - gt0) if not args.e2e_wall else (
            gtok / sum(greedy_e2e_walls) if greedy_e2e_walls else None
        )

    o0_archive_fail = False
    o0_safety_fail = False
    o0_arch_status = ARCHIVE_GATE_NOT_RUN
    o0_byte_exact = None
    o0_safety = None
    n_archive_covered = n_fingerprint = n_exact = n_len_boundary = n_near_tie = n_hard = 0
    if args.check_greedy_bytes:
        for i, p in enumerate(prompts):
            g = greedy_seqs[i]
            emitted = results[i]["emitted_tokens"]
            pid = str(p["id"])
            row: Dict = {"prompt_id": pid, "image": p["image"]}

            # --- Verdict 1: archive reproducibility (byte-exact vs archive) ---
            if pid in o0_archive_map:
                n_archive_covered += 1
                row["spec_verdict"] = "covered"
                if o0_archive_triggers_fail(emitted, o0_archive_map[pid]):
                    cls_a = classify_o0_vs_ref(emitted, o0_archive_map[pid])
                    rec = {
                        "prompt_id": pid, "image": p["image"],
                        "verdict": "archive_reproducibility",
                        "kind": cls_a["kind"], **(cls_a["div"] or {}),
                    }
                    o0_spec_fails.append(rec)
                    row["spec_kind"] = cls_a["kind"]
                    print(f"[O0 ARCHIVE FAIL] id={pid} kind={cls_a['kind']} "
                          f"pos={rec.get('pos')}")
                else:
                    row["spec_kind"] = "match"
            else:
                n_fingerprint += 1
                row["spec_verdict"] = "uncovered"
                row["spec_kind"] = None

            # --- Verdict 2: vs independent greedy (safety + byte-exact counts) ---
            div_g = compare_token_sequences(emitted, g)
            probe = None
            if div_g is not None and div_g["pos"] < min(div_g["spec_len"], div_g["greedy_len"]):
                probe = greedy_candidate_probe_at(
                    base, processor, p["question"], images_dir / p["image"],
                    g[: div_g["pos"]], div_g["spec_tok"],
                )
            gap_g = None if probe is None else probe.get("top2_logit_gap")
            cls_g = classify_o0_vs_ref(
                emitted, g, top2_gap=gap_g,
                greedy_top1=None if probe is None else probe.get("greedy_top1"),
                greedy_top2=None if probe is None else probe.get("greedy_top2"),
                gap_spec=None if probe is None else probe.get("gap_spec"),
                spec_rank=None if probe is None else probe.get("spec_rank"),
            )
            kind_g = cls_g["kind"]
            row["greedy_kind"] = kind_g
            row["top2_logit_gap"] = gap_g
            row["greedy_top1"] = cls_g.get("greedy_top1")
            row["greedy_top2"] = cls_g.get("greedy_top2")
            row["gap_spec"] = cls_g.get("gap_spec")
            row["spec_rank"] = cls_g.get("spec_rank")
            if cls_g["div"] is not None:
                row.update({k: cls_g["div"][k] for k in
                            ("pos", "spec_tok", "greedy_tok", "spec_len", "greedy_len")
                            if k in cls_g["div"]})
            # Full greedy prefix identity (this version on; not backfilled):
            # binds the row to the exact prefix, unlike the ±5-token window.
            prefix_toks = g if div_g is None else g[: div_g["pos"]]
            row["greedy_prefix_len"] = len(prefix_toks)
            row["greedy_prefix_sha256"] = token_seq_sha256(prefix_toks)
            row["greedy_full_sha256"] = token_seq_sha256(g)
            cand_fields = {
                "top2_logit_gap": gap_g,
                "greedy_top1": cls_g.get("greedy_top1"),
                "greedy_top2": cls_g.get("greedy_top2"),
                "gap_spec": cls_g.get("gap_spec"),
                "spec_rank": cls_g.get("spec_rank"),
                "greedy_prefix_len": row["greedy_prefix_len"],
                "greedy_prefix_sha256": row["greedy_prefix_sha256"],
                "greedy_full_sha256": row["greedy_full_sha256"],
            }
            if kind_g == "match":
                n_exact += 1
                o0_exact_ids.append(pid)
            elif kind_g == "near_tie":
                n_near_tie += 1
                o0_near_ties.append({
                    "prompt_id": pid, "image": p["image"],
                    "verdict": "greedy_numerical_safety", "kind": "near_tie",
                    **cand_fields, **(cls_g["div"] or {}),
                })
                print(f"[O0 near_tie] id={pid} pos={row.get('pos')} "
                      f"gap_spec={cand_fields['gap_spec']} rank={cand_fields['spec_rank']} "
                      f"top2={cand_fields['greedy_top2']} "
                      f"(not byte-exact vs greedy; safety non-fail)")
            elif kind_g == "len_boundary":
                n_len_boundary += 1
                rec = {
                    "prompt_id": pid, "image": p["image"],
                    "verdict": "greedy_numerical_safety", "kind": "len_boundary",
                    **cand_fields, **(cls_g["div"] or {}),
                }
                o0_greedy_safety_fails.append(rec)
                print(f"[O0 SAFETY FAIL len_boundary] id={pid} "
                      f"pos={rec.get('pos')}")
            else:  # hard
                n_hard += 1
                rec = {
                    "prompt_id": pid, "image": p["image"],
                    "verdict": "greedy_numerical_safety", "kind": "hard",
                    **cand_fields, **(cls_g["div"] or {}),
                }
                o0_greedy_safety_fails.append(rec)
                print(f"[O0 SAFETY FAIL hard] id={pid} pos={rec.get('pos')} "
                      f"gap_spec={cand_fields['gap_spec']} rank={cand_fields['spec_rank']}")
            o0_per_prompt.append(row)

        n = len(prompts)
        o0_arch_status = archive_gate_status(
            archive_provided, n_archive_covered, n, len(o0_spec_fails),
        )
        covered_subset_pass = (len(o0_spec_fails) == 0) if n_archive_covered > 0 else None
        o0_archive_fail = (o0_arch_status == ARCHIVE_GATE_FAIL)
        o0_byte_exact = greedy_byte_exact_pass(n_exact, n)
        o0_safety = greedy_numerical_safety_pass(n_len_boundary, n_hard)
        o0_safety_fail = not o0_safety
        print(f"\n=== O0 DUAL VERDICT (band={O0_KERNEL_BAND}; Tier D) ===")
        print(f"  n_archive_covered={n_archive_covered}  n_fingerprint={n_fingerprint}")
        print(f"  n_exact={n_exact}  n_len_boundary={n_len_boundary}  "
              f"n_near_tie={n_near_tie}  n_hard={n_hard}")
        print(f"  archive_gate: {o0_arch_status} "
              f"(fails={len(o0_spec_fails)}; "
              f"covered_subset_pass={covered_subset_pass})")
        print(f"  greedy_byte_exact_pass: {o0_byte_exact} "
              f"(n_exact==n_prompts; expected FALSE if near_tie>0)")
        print(f"  greedy_numerical_safety_pass: {o0_safety} "
              f"(fails={len(o0_greedy_safety_fails)}; near_tie={n_near_tie} calibrated)")
        print(f"  claim: {O0_CLAIM_OFFICIAL}")
        print("  near_tie rule: candidate-specific gap-only "
              "(0<=gap_spec<=band); top1-top2 gap and spec_rank are diagnostics")
        dump = {
            "protocol": "dual_verdict_tier_d",
            "runner": "v2_hardcap",
            "kernel_band": O0_KERNEL_BAND,
            "near_tie_rule": "candidate_specific_gap_only",
            "fingerprint": o0_run_fingerprint(args),
            "archive_path": args.o0_archive,
            "n_prompts": n,
            "n_archive_covered": n_archive_covered,
            "n_fingerprint": n_fingerprint,
            "n_exact": n_exact,
            "n_len_boundary": n_len_boundary,
            "n_near_tie": n_near_tie,
            "n_hard": n_hard,
            "archive_gate_status": o0_arch_status,
            "archive_covered_subset_pass": covered_subset_pass,
            "greedy_byte_exact_pass": o0_byte_exact,
            "greedy_numerical_safety_pass": o0_safety,
            # Sole allowed deprecated alias for numerical safety (not byte-exact)
            "legacy_numerical_safety_pass_deprecated": o0_safety,
            "archive_reproducibility_pass": o0_arch_status == ARCHIVE_GATE_PASS,
            "official_claim": O0_CLAIM_OFFICIAL,
            "spec_fails": o0_spec_fails,
            "greedy_safety_fails": o0_greedy_safety_fails,
            "near_ties": o0_near_ties,
            "per_prompt": o0_per_prompt,
        }
        out_div = Path(args.out).with_suffix(".o0_report.json")
        out_div.parent.mkdir(parents=True, exist_ok=True)
        with open(out_div, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2, ensure_ascii=False)
        print(f"[O0] wrote {out_div}")

        def _write_arch(path: str, rows: List[Dict], archive_kind: str) -> None:
            arch_out = {
                "source_out": args.out,
                "archive_kind": archive_kind,
                "o0_provenance": {
                    "job_id": os.environ.get("PBS_JOBID"),
                    "source_out": args.out,
                    "fanout": args.fanout,
                    "max_nodes": args.max_nodes,
                    "ckpt": args.ckpt,
                    "runner": "v2_hardcap",
                    "archive_kind": archive_kind,
                },
                "n_prompts": len(rows),
                "per_prompt_results": rows,
            }
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(arch_out, f, indent=2)
            side = Path(str(Path(path).resolve()) + ".o0_prov.json")
            side.write_text(json.dumps({
                "job_id": os.environ.get("PBS_JOBID"),
                "source_out": args.out,
                "sha256": sha256_file(Path(path)),
                "archive_path": str(Path(path).resolve()),
                "runner": "v2_hardcap",
                "archive_kind": archive_kind,
            }, indent=2))
            print(f"[O0] wrote {archive_kind} archive {path} + {side.name}")

        # Spec archive write: only when PASS (reconfirm) or NOT_RUN (fresh)
        if args.o0_write_archive:
            if o0_arch_status in (ARCHIVE_GATE_PASS, ARCHIVE_GATE_NOT_RUN):
                _write_arch(
                    args.o0_write_archive,
                    [{"id": r["id"], "emitted_tokens": r["emitted_tokens"]}
                     for r in results],
                    "archive_reproducibility",
                )
            else:
                print(f"[O0] skip archive write: archive_gate={o0_arch_status}")

        # Greedy archive: ONLY byte-exact prompts, require numerical safety
        if args.o0_write_greedy_archive:
            if o0_safety_fail:
                print("[O0] skip greedy archive: numerical_safety FAIL "
                      "(len_boundary/hard present)")
            else:
                exact_set = set(o0_exact_ids)
                rows_g = [
                    {"id": r["id"], "emitted_tokens": r["emitted_tokens"]}
                    for r in results if str(r["id"]) in exact_set
                ]
                _write_arch(args.o0_write_greedy_archive, rows_g, "greedy_byte_exact")
                print(f"[O0] greedy byte-exact archive n_exact={len(rows_g)} "
                      f"(excluded near_tie={n_near_tie}; "
                      f"byte_exact_pass={o0_byte_exact})")

    o0_gate_failed = o0_archive_fail or o0_safety_fail

    chain_tok_s = chain_sigma = None
    if args.measure_chain:
        from scripts.eval_acceptance import run_one_prompt as chain_run
        ct0 = time.time(); ctok = 0; csig = []
        for p in prompts:
            cr = chain_run(base, head, processor, p, images_dir, args.max_new_tokens, eos_id)
            ctok += cr["total_emitted"]; csig.append(cr["sigma"])
        chain_tok_s = ctok / (time.time() - ct0)
        chain_sigma = statistics.mean(csig)

    sigmas = [r["sigma"] for r in results]
    total_emit = sum(r["total_emitted"] for r in results)
    total_rounds = sum(r["rounds"] for r in results)
    K = head.num_heads
    depth_accept = Counter()
    for r in results:
        for d, c in r["depth_accept_counts"].items():
            depth_accept[int(d)] += c
    tree_tok_s = total_emit / decode_wall
    # Dual σ / N̄ accounting (prompt_mean = legacy; round_pooled for matched-compute)
    sigma_prompt_mean = statistics.mean(sigmas) if sigmas else 0.0
    sigma_round_pooled = total_emit / max(1, total_rounds)
    nbar_prompt_mean = statistics.mean([r["mean_width"] for r in results]) if results else 0.0
    width_round_sum = sum(r["mean_width"] * r["rounds"] for r in results)
    nbar_round_pooled = width_round_sum / max(1, total_rounds)
    # Official e2e paired speedup (when --e2e-wall)
    paired_sp = [r.get("paired_speedup") for r in results if r.get("paired_speedup") is not None]
    paired_mean = paired_se = None
    if paired_sp:
        paired_mean = statistics.mean(paired_sp)
        paired_se = (
            statistics.stdev(paired_sp) / (len(paired_sp) ** 0.5) if len(paired_sp) > 1 else 0.0
        )
    e2e_spec_walls = [r.get("e2e_wall_s") for r in results if r.get("e2e_wall_s") is not None]
    e2e_tree_tok_s = (
        total_emit / sum(e2e_spec_walls) if e2e_spec_walls and sum(e2e_spec_walls) > 0 else None
    )
    agg = {
        "mode": "tree", "n_prompts": len(results), "fanout": args.fanout,
        "max_nodes": args.max_nodes, "depth1_floor": not args.no_depth1_floor,
        "reorg": args.reorg,
        "sigma_mean": sigma_prompt_mean,  # alias = prompt_mean (primary cite)
        "sigma_prompt_mean": sigma_prompt_mean,
        "sigma_round_pooled": sigma_round_pooled,
        "sigma_median": statistics.median(sigmas) if sigmas else None,
        "sigma_p25": statistics.quantiles(sigmas, n=4)[0] if len(sigmas) >= 4 else None,
        "sigma_p75": statistics.quantiles(sigmas, n=4)[2] if len(sigmas) >= 4 else None,
        "mean_tree_width": nbar_prompt_mean,  # alias = prompt_mean
        "nbar_prompt_mean": nbar_prompt_mean,
        "nbar_round_pooled": nbar_round_pooled,
        "mean_depth_widths": [statistics.mean(c) for c in zip(*[r["mean_depth_widths"] for r in results])] if results else [],
        "depth_accept_counts": dict(sorted(depth_accept.items())),
        "per_position_accept_rate": [
            sum(int(a > k) for r in results for a in r["accept_log"]) / max(1, total_rounds)
            for k in range(K)
        ],
        "total_rounds": total_rounds, "total_tokens_emitted": total_emit,
        "decode_wall_s": decode_wall, "tree_tok_per_s": tree_tok_s,
        "greedy_tok_per_s": greedy_tok_s,
        "chain_tok_per_s": chain_tok_s, "chain_sigma_measured": chain_sigma,
        "net_speedup_vs_greedy": (tree_tok_s / greedy_tok_s) if greedy_tok_s else None,
        "tree_speedup_vs_chain": (tree_tok_s / chain_tok_s) if chain_tok_s else None,
        "chain_speedup_vs_greedy": (chain_tok_s / greedy_tok_s) if (chain_tok_s and greedy_tok_s) else None,
        "timing_protocol": "e2e_wall" if args.e2e_wall else "legacy_segment_aggregate",
        "e2e_wall": bool(args.e2e_wall),
        "runner": "v2_hardcap",
        "e2e_tree_tok_per_s": e2e_tree_tok_s,
        "paired_speedup_mean": paired_mean,
        "paired_speedup_se": paired_se,
        "paired_speedup_n": len(paired_sp) if paired_sp else 0,
        "sigma_chain_v1": 1.95,
        "ordered_manifest": args.ordered_manifest,
        "check_greedy_bytes": args.check_greedy_bytes,
        "o0_archive": args.o0_archive,
        "o0_archive_sha256": (o0_archive_meta or {}).get("sha256") if args.check_greedy_bytes else None,
        "o0_archive_job_id": (o0_archive_meta or {}).get("job_id") if args.check_greedy_bytes else None,
        "o0_kernel_band": O0_KERNEL_BAND if args.check_greedy_bytes else None,
        "tree_builder": args.tree_builder,
        "o0_protocol": "dual_verdict_tier_d" if args.check_greedy_bytes else None,
        "o0_n_archive_covered": n_archive_covered if args.check_greedy_bytes else None,
        "o0_n_fingerprint": n_fingerprint if args.check_greedy_bytes else None,
        "o0_n_exact": n_exact if args.check_greedy_bytes else None,
        "o0_n_len_boundary": n_len_boundary if args.check_greedy_bytes else None,
        "o0_n_near_tie": n_near_tie if args.check_greedy_bytes else None,
        "o0_n_hard": n_hard if args.check_greedy_bytes else None,
        "o0_archive_gate_status": o0_arch_status if args.check_greedy_bytes else None,
        "o0_greedy_byte_exact_pass": o0_byte_exact if args.check_greedy_bytes else None,
        "o0_greedy_numerical_safety_pass": o0_safety if args.check_greedy_bytes else None,
        "o0_legacy_numerical_safety_pass_deprecated": (
            o0_safety if args.check_greedy_bytes else None),
        "o0_archive_reproducibility_pass": (
            o0_arch_status == ARCHIVE_GATE_PASS) if args.check_greedy_bytes else None,
        "per_prompt_results": results,
        "max_pixels": args.max_pixels,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    print(f"[mem] peak_alloc={agg['peak_alloc_gib']:.3f} GiB  "
          f"peak_reserved={agg['peak_reserved_gib']:.3f} GiB")
    if args.profile:
        comp = {k: sum(r["profile_s"][k] for r in results)
                for k in ("head", "verify", "reorg", "bonus", "other")}
        tot = sum(comp.values())
        per_round = {k: v / max(1, total_rounds) for k, v in comp.items()}
        base_unit_s = (1.0 / greedy_tok_s) if greedy_tok_s else None  # ~single-token base fwd
        agg["profile_total_s"] = comp
        agg["profile_per_round_ms"] = {k: 1000 * v for k, v in per_round.items()}
        agg["profile_frac"] = {k: v / max(1e-9, tot) for k, v in comp.items()}
        agg["base_unit_s"] = base_unit_s
        agg["bonus_in_base_units"] = (per_round["bonus"] / base_unit_s) if base_unit_s else None
        agg["verify_in_base_units"] = (per_round["verify"] / base_unit_s) if base_unit_s else None
        agg["head_in_base_units"] = (per_round["head"] / base_unit_s) if base_unit_s else None
    print("\n=== TREE AGGREGATE ===")
    print(f"  σ prompt_mean = {agg['sigma_prompt_mean']:.3f}  "
          f"(round_pooled = {agg['sigma_round_pooled']:.3f}; chain v1 = 1.95)")
    print(f"  N̄ prompt_mean = {agg['nbar_prompt_mean']:.2f}  "
          f"(round_pooled = {agg['nbar_round_pooled']:.2f}; "
          f"per-depth {agg['mean_depth_widths']})")
    print(f"  per-position accept = {[f'{x:.3f}' for x in agg['per_position_accept_rate']]}")
    print(f"  depth_accept_counts = {agg['depth_accept_counts']}")
    print(f"  tree   tok/s = {tree_tok_s:.2f}")
    if greedy_tok_s:
        print(f"  greedy tok/s = {greedy_tok_s:.2f}   tree net_speedup vs greedy = {agg['net_speedup_vs_greedy']:.3f}")
    if args.e2e_wall and paired_mean is not None:
        print(f"  [OFFICIAL e2e-wall] paired speedup = {paired_mean:.4f} ± {paired_se:.4f} "
              f"(n={len(paired_sp)}; ratio = greedy_wall/spec_wall)")
        if e2e_tree_tok_s is not None:
            print(f"  [e2e-wall] tree tok/s (sum emit / sum e2e walls) = {e2e_tree_tok_s:.2f}")
    if chain_tok_s:
        print(f"  chain  tok/s = {chain_tok_s:.2f} (σ={chain_sigma:.3f})   "
              f"chain speedup vs greedy = {agg['chain_speedup_vs_greedy']}   "
              f"tree speedup vs chain = {agg['tree_speedup_vs_chain']}")
    if args.profile:
        print("\n  --- per-round wall-time breakdown (3a) ---")
        for k in ("head", "verify", "reorg", "bonus", "other"):
            print(f"    {k:7s}: {agg['profile_per_round_ms'][k]:7.2f} ms/round "
                  f"({100*agg['profile_frac'][k]:5.1f}%)")
        if agg.get("base_unit_s"):
            print(f"    base-unit (1-tok greedy fwd) = {1000*agg['base_unit_s']:.2f} ms")
            print(f"    bonus_fwd  = {agg['bonus_in_base_units']:.2f} base-units  "
                  f"(removable target)")
            print(f"    verify_fwd = {agg['verify_in_base_units']:.2f} base-units  "
                  f"(memory-bound: ~1 regardless of N)")
            print(f"    head_fwd   = {agg['head_in_base_units']:.2f} base-units")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print(f"[save] -> {args.out}")
    if args.check_greedy_bytes:
        return _combine_o0_exit(o0_arch_status, archive_provided, o0_safety_fail)
    return 0


NEAR_TIE_MARGIN = 0.5   # fp16 SDPA-kernel noise scale (probe CHECK 1 measured ~0.1)


def _greedy_e2e(base, processor, prompts, images_dir, eos_id, args) -> int:
    """Official e2e-wall greedy baseline (warmup + per-prompt CUDA-synced walls)."""
    images_dir = Path(images_dir)
    if prompts:
        print("[e2e-wall] greedy warmup prompt 0 (timing discarded)…")
        vanilla_greedy(base, processor, prompts[0]["question"],
                       images_dir / prompts[0]["image"], args.max_new_tokens, eos_id)
        _cuda_sync()
    results = []
    for i, p in enumerate(prompts):
        g, wall = _e2e_wall_call(
            lambda p=p: vanilla_greedy(
                base, processor, p["question"], images_dir / p["image"],
                args.max_new_tokens, eos_id,
            )
        )
        results.append({
            "id": p["id"], "image": p["image"],
            "total_emitted": len(g), "emitted_tokens": g,
            "e2e_wall_s": wall, "hit_eos": bool(g and g[-1] == eos_id),
        })
        print(f"[greedy_e2e {i+1:>3}/{len(prompts)}] emit={len(g)} wall={wall:.3f}s")
    total_emit = sum(r["total_emitted"] for r in results)
    total_wall = sum(r["e2e_wall_s"] for r in results)
    agg = {
        "mode": "greedy_e2e",
        "n_prompts": len(results),
        "timing_protocol": "e2e_wall",
        "e2e_wall": True,
        "total_tokens_emitted": total_emit,
        "total_e2e_wall_s": total_wall,
        "greedy_tok_per_s": (total_emit / total_wall) if total_wall > 0 else None,
        "per_prompt_results": results,
        "max_pixels": args.max_pixels,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    print(f"\n=== GREEDY E2E === tok/s={agg['greedy_tok_per_s']:.3f}  "
          f"wall={total_wall:.1f}s  emit={total_emit}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[save] -> {args.out}")
    return 0


@torch.no_grad()
def _validate_reorg(base, head, processor, prompts, images_dir, eos_id, args) -> int:
    """Stepwise argmax validation of the gather-maintained KV cache.

    RETIRED the old gather==safe byte-identical assertion: post-Q2, `safe` is the
    drifting path (its recompute uses the is_causal kernel vs the verify's masked
    kernel), so a long decode eventually hits an fp16 near-tie and the byte
    compare FALSELY reports "keep safe". gather correctness is established by the
    regression gate (fanout=[1,1,1] ≡ chain, 5/5) + the geometric argument.

    Instead, for WIDE trees (which the degenerate gate doesn't exercise), verify
    that the gather cache stays consistent with a clean from-scratch forward:
    every round, recompute [prefix + emitted-so-far] from scratch and check that
    its next-token argmax equals base_pred_root carried by the gather path. A
    mismatch is tolerated ONLY if the from-scratch top-2 margin is a near-tie
    (< NEAR_TIE_MARGIN) — i.e. fp16, not a cache-corruption bug.
    """
    from decode.tree import build_tree, build_mask_and_positions, tree_tokens, accept, reorg_kv_gather
    print("\n=== VALIDATE REORG (gather): stepwise argmax vs from-scratch ===")
    dtype = base.dtype
    n = min(len(prompts), args.n_prompts, 5)
    total_checks = clean = near_tie = hard_fail = 0
    for i, p in enumerate(prompts[:n]):
        inputs = make_image_inputs(processor, p["question"], images_dir / p["image"], device="cuda:0")
        out = base(**inputs, use_cache=True, output_hidden_states=True)
        past_kv = out.past_key_values
        P = past_kv.get_seq_length()
        h_t = out.hidden_states[-1][0, -1, :].clone()
        base_pred_root = argmax_masked(out.logits[0, -1, :])
        del out
        emitted: List[int] = []
        p_fail = 0
        while len(emitted) < args.max_new_tokens:
            # from-scratch reference: forward [prefix + emitted] fresh, check next argmax
            with torch.no_grad():
                ref = base(**inputs, use_cache=True) if not emitted else base(
                    input_ids=torch.tensor([emitted], device="cuda:0"),
                    past_key_values=base(**inputs, use_cache=True).past_key_values)
            ref_logits = ref.logits[0, -1, :]
            ref_argmax = argmax_masked(ref_logits)
            total_checks += 1
            if ref_argmax == base_pred_root:
                clean += 1
            else:
                m = _margin_top2(ref_logits)
                if m < NEAR_TIE_MARGIN:
                    near_tie += 1
                else:
                    hard_fail += 1; p_fail += 1
                    print(f"    [HARD FAIL] {p['image']} @emit{len(emitted)}: "
                          f"gather={base_pred_root} ref={ref_argmax} margin={m:.3e}")
            del ref
            # advance one tree round (gather)
            cont_base = continuation_base(base, P)
            nodes = build_tree(head(h_t.view(1, 1, -1).half()), args.fanout, args.max_nodes,
                               depth1_floor=not args.no_depth1_floor)
            mask, pos = build_mask_and_positions(nodes, P, cont_base, dtype, "cuda:0")
            v_out = base(input_ids=tree_tokens(nodes, "cuda:0"), attention_mask=mask,
                         past_key_values=past_kv, position_ids=pos, use_cache=True)
            accepted, _, bonus, _ = accept(nodes, v_out.logits[0], base_pred_root)
            emitted_round = [nodes[j].token for j in accepted] + [bonus]
            if eos_id in emitted_round:
                emitted.extend(emitted_round[:emitted_round.index(eos_id) + 1]); break
            emitted.extend(emitted_round)
            reorg_kv_gather(v_out.past_key_values, P, accepted, "cuda:0")
            past_kv = v_out.past_key_values
            del v_out
            P = past_kv.get_seq_length()
            cb = continuation_base(base, P)
            b_out = base(input_ids=torch.tensor([[bonus]], device="cuda:0"),
                         past_key_values=past_kv, position_ids=torch.tensor([[cb]], device="cuda:0"),
                         use_cache=True, output_hidden_states=True)
            past_kv = b_out.past_key_values
            h_t = b_out.hidden_states[-1][0, -1, :].clone()
            base_pred_root = argmax_masked(b_out.logits[0, -1, :])
            P = past_kv.get_seq_length()
            del b_out
        print(f"[val {i+1}/{n}] {p['image']}  checks={len(emitted)}  hard_fails={p_fail}")
    print(f"\n  stepwise argmax: {clean}/{total_checks} clean, {near_tie} near-tie (fp16 OK), "
          f"{hard_fail} HARD FAIL")
    print(f"  GATHER CACHE CONSISTENT: {'YES ✓' if hard_fail == 0 else 'NO ✗ (cache bug)'}")
    return 0 if hard_fail == 0 else 1


def _margin_top2(logits_1d) -> float:
    from decode.common import mask_phantom_
    t2 = mask_phantom_(logits_1d).float().topk(2).values
    return (t2[0] - t2[1]).item()


@torch.no_grad()
def _divergence_margin(base, inputs, ref_tokens, d, device="cuda:0") -> float:
    """from-scratch forward of [prefix + ref[:d]] -> base top-2 margin at position d.

    < NEAR_TIE_MARGIN => the divergence is an fp16 near-tie (kernel noise), not a bug.
    """
    pref = base(**inputs, use_cache=True)
    if d == 0:
        logits = pref.logits[0, -1, :]
    else:
        o = base(input_ids=torch.tensor([ref_tokens[:d]], device=device),
                 past_key_values=pref.past_key_values, use_cache=True)
        logits = o.logits[0, -1, :]
    return _margin_top2(logits)


@torch.no_grad()
def _classify_vs_greedy(base, processor, prompt, images_dir, device, ref, emitted):
    """First divergence of `emitted` vs vanilla-greedy `ref`; classify near-tie/hard."""
    m = min(len(ref), len(emitted))
    d = next((j for j in range(m) if ref[j] != emitted[j]), None)
    if d is None:
        return ("match", m, m, None)
    inputs = make_image_inputs(processor, prompt["question"], images_dir / prompt["image"], device)
    margin = _divergence_margin(base, inputs, ref, d, device)
    kind = "near_tie" if margin < NEAR_TIE_MARGIN else "HARD"
    return (kind, d, m, margin)


KERNEL_BAND = 0.15   # fp16 masked-vs-causal kernel flips only near-ties (~0.1);
                     # a divergence at margin > this is NOT attributable to the kernel.


@torch.no_grad()
def _greedy_agreement(base, head, processor, prompts, images_dir, eos_id, args) -> int:
    """Disambiguate folded's greedy divergence: fp16 kernel noise vs real bug.

    The discriminator is NOT an absolute margin threshold (the kernel fp16 diff is
    only ~0.1 and cannot flip a margin-0.3..0.5 decision, so margin<0.5 would mask
    a real bug). The discriminator is the BASELINE-tree control: folded's
    divergence profile vs vanilla greedy must ≈ baseline-tree's profile. Only
    "folded diverges where baseline also diverges (same points/margins)" is fp16
    noise. A folded-UNIQUE divergence (baseline matched / diverged later) at
    margin > KERNEL_BAND is a folded bug.

    Plus a plumbing health check (orthogonal to correctness — catches a wiring bug
    that silently suppresses σ): head_0(h_anchor) top1 should re-predict the known
    bonus at ≈0.871 (baseline depth-1 accept). And folded's measured head_1/head_2
    per-level accept should ≈ baseline [_, 0.384, 0.117].
    """
    print("\n=== GREEDY AGREEMENT (baseline tree & folded vs vanilla greedy) ===")
    n = min(len(prompts), args.n_prompts)
    dev = "cuda:0"
    K = head.num_heads
    BASE_REF_PPA = [0.871, 0.384, 0.117]
    rec = {"baseline": [], "folded": []}            # per-prompt (kind, d, m, margin)
    base_ppa_acc = [0.0] * K; base_rounds = 0
    fold_ppa_acc = [0.0] * K; fold_rounds = 0; head0_hit_rounds = 0
    h0_hits_by_depth: Counter = Counter(); h0_rounds_by_depth: Counter = Counter()
    headk_hits_by_depth = {k: Counter() for k in range(K)}   # real bug detector (head_1/head_2)
    for i, p in enumerate(prompts[:n]):
        ref = vanilla_greedy(base, processor, p["question"], images_dir / p["image"],
                             args.max_new_tokens, eos_id)
        rb = run_one_prompt_tree(base, head, processor, p, images_dir, args.max_new_tokens,
                                 eos_id, args.fanout, args.max_nodes, True, "gather")
        rf = run_one_prompt_tree_folded(base, head, processor, p, images_dir, args.max_new_tokens,
                                        eos_id, args.fanout, args.max_nodes,
                                        skip_head0_lm_head=False)   # keep head0 diagnostic
        # baseline per-position accept (recompute from accept_log)
        for k in range(K):
            base_ppa_acc[k] += sum(int(a > k) for a in rb["accept_log"])
            fold_ppa_acc[k] += rf["per_position_accept"][k] * rf["rounds"]
        base_rounds += rb["rounds"]; fold_rounds += rf["rounds"]
        head0_hit_rounds += rf["head0_top1_rate"] * rf["rounds"]
        for d, rr in rf["rounds_by_anchor_depth"].items():
            h0_rounds_by_depth[d] += rr
            h0_hits_by_depth[d] += round(rf["head0_rate_by_anchor_depth"][d] * rr)
            for k in range(K):
                headk_hits_by_depth[k][d] += round(rf["headk_accept_by_anchor_depth"][k][d] * rr)
        for name, em in (("baseline", rb["emitted_tokens"]), ("folded", rf["emitted_tokens"])):
            kind, d, m, margin = _classify_vs_greedy(base, processor, p, images_dir, dev, ref, em)
            rec[name].append((kind, d, m, margin))
            tag = "" if kind == "match" else f"  diverge@{d}/{m} margin={margin:.3e}"
            print(f"[{i+1:>2}/{n}] {name:8s} {p['image']}  ref={len(ref)} emit={len(em)}{tag}")

    def _counts(rs):
        return Counter(k for k, _, _, _ in rs)

    def _margins(rs):
        return sorted(mg for k, _, _, mg in rs if mg is not None)

    bc, fc = _counts(rec["baseline"]), _counts(rec["folded"])
    bm, fm = _margins(rec["baseline"]), _margins(rec["folded"])

    # folded-unique divergences: folded diverges where baseline matched OR earlier
    folded_unique = []
    for (kb, db, _, _), (kf, df, _, mgf) in zip(rec["baseline"], rec["folded"]):
        if kf != "match":
            base_div_at_or_before = (kb != "match" and db is not None and db <= df)
            if not base_div_at_or_before and mgf is not None and mgf > KERNEL_BAND:
                folded_unique.append((df, mgf))

    print("\n  --- divergence profile (folded must ≈ baseline) ---")
    for name, c, mg in (("baseline", bc, bm), ("folded", fc, fm)):
        mstr = (f"margins[min={mg[0]:.3e} med={statistics.median(mg):.3e} max={mg[-1]:.3e}]"
                if mg else "margins[none]")
        print(f"  {name:8s}: match={c['match']} diverge={sum(c.values())-c['match']}  {mstr}")
    print(f"  folded-UNIQUE divergences (baseline matched, margin>{KERNEL_BAND}): "
          f"{len(folded_unique)}  {folded_unique if folded_unique else ''}")

    base_ppa = [x / max(1, base_rounds) for x in base_ppa_acc]
    fold_ppa = [x / max(1, fold_rounds) for x in fold_ppa_acc]
    head0_rate = head0_hit_rounds / max(1, fold_rounds)

    # REAL bug detector: head_1/head_2 accept by anchor depth. Collapse ONLY in the
    # deep bucket (anchor_depth>=2) vs the shallow buckets = real wiring bug; uniform
    # ≈ baseline = expected structural cost. (head_0 is NOT a gate — unused in the
    # folded tree, and its 'rate' is a different task than baseline head_0.)
    def _hk_at(k, depths):
        hits = sum(headk_hits_by_depth[k][d] for d in depths)
        rr = sum(h0_rounds_by_depth[d] for d in depths)
        return (hits / rr if rr else float("nan")), rr
    shallow = [d for d in h0_rounds_by_depth if d <= 1]
    deep = [d for d in h0_rounds_by_depth if d >= 2]
    print("\n  --- head_1/head_2 accept by anchor depth (REAL bug detector) ---")
    print(f"  baseline per-position accept : {[f'{x:.3f}' for x in base_ppa]}")
    print(f"  folded   per-position accept : {[f'{x:.3f}' for x in fold_ppa]}  "
          f"([0]=bonus root; [1]=head_1 vs base {BASE_REF_PPA[1]}; [2]=head_2 vs base {BASE_REF_PPA[2]})")
    for k in (1, 2):
        sh, shn = _hk_at(k, shallow); dp, dpn = _hk_at(k, deep)
        print(f"  head_{k} accept  shallow(anchor≤1)={sh:.3f} (n={shn})   "
              f"deep(anchor≥2)={dp:.3f} (n={dpn})")
    print(f"  [observation only] head0_rate={head0_rate:.3f} "
          f"(NOT comparable to baseline 0.85: head_0 unused in tree, different task)")

    # gate: deep bucket must not collapse vs shallow for head_1 (the σ workhorse)
    h1_sh, _ = _hk_at(1, shallow); h1_dp, h1_dpn = _hk_at(1, deep)
    deep_collapse = (h1_dpn >= 20 and h1_dp == h1_dp and h1_sh == h1_sh
                     and h1_dp < 0.5 * h1_sh)   # deep < half of shallow = suspicious
    corr_ok = (len(folded_unique) == 0)
    plumb_ok = (not deep_collapse)
    sigma_est = 1.0 + sum(fold_ppa[1:])   # E[accept_len] = 1 (root) + head_1 + head_2 + ...
    print("\n  --- verdict ---")
    print(f"  correctness: {'fp16 kernel noise (folded ≈ baseline) ✓' if corr_ok else 'FOLDED-SPECIFIC divergence ✗ (investigate)'}")
    print(f"  bug-detector: {'deep bucket OK (head_1/head_2 ≈ shallow ≈ baseline) ✓' if plumb_ok else 'DEEP-BUCKET COLLAPSE ✗ (real bug, fix first)'}"
          f"  — σ≈{sigma_est:.3f} (1+head_1+head_2) is real structural cost")
    adopt = corr_ok and plumb_ok
    print(f"\n  ADOPT FOLD as speed-optimized path? {'YES ✓' if adopt else 'NO ✗'}  "
          f"(keep baseline tree as high-σ tradeoff point either way)")
    out = {"n_prompts": n,
           "baseline_counts": dict(bc), "folded_counts": dict(fc),
           "baseline_margins": bm, "folded_margins": fm,
           "folded_unique_divergences": folded_unique,
           "head0_rate_observation": head0_rate,
           "head1_shallow": _hk_at(1, shallow)[0], "head1_deep": _hk_at(1, deep)[0],
           "head2_shallow": _hk_at(2, shallow)[0], "head2_deep": _hk_at(2, deep)[0],
           "baseline_per_position_accept": base_ppa, "folded_per_position_accept": fold_ppa,
           "correctness_ok": corr_ok, "deep_bucket_ok": plumb_ok, "adopt_fold": adopt}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[save] -> {args.out}")
    return 0 if adopt else 1


def _foldbonus_ab(base, head, processor, prompts, images_dir, eos_id, args) -> int:
    """(3c) A/B: gather baseline vs fold-bonus, same prompts/GPU/process.

    Decision metric = tok/s. Adopt fold only if tok/s(folded) > tok/s(baseline).
    Also report σ change and a correctness gate: folded greedy output must match
    vanilla_greedy argmax (common-prefix; rare fp16 near-tie divergence allowed).
    """
    print(f"\n=== FOLD-BONUS A/B (fanout={args.fanout} max_nodes={args.max_nodes}) ===")
    n = min(len(prompts), args.n_prompts)
    sel = prompts[:n]

    # baseline gather tree
    t0 = time.time(); base_emit = 0; base_sig = []
    for p in sel:
        r = run_one_prompt_tree(base, head, processor, p, images_dir, args.max_new_tokens,
                                eos_id, args.fanout, args.max_nodes, True, "gather")
        base_emit += r["total_emitted"]; base_sig.append(r["sigma"])
    base_wall = time.time() - t0
    base_toks = base_emit / base_wall

    # folded
    t0 = time.time(); fold_emit = 0; fold_sig = []; fold_out = []
    for p in sel:
        r = run_one_prompt_tree_folded(base, head, processor, p, images_dir, args.max_new_tokens,
                                       eos_id, args.fanout, args.max_nodes)
        fold_emit += r["total_emitted"]; fold_sig.append(r["sigma"]); fold_out.append(r["emitted_tokens"])
    fold_wall = time.time() - t0
    fold_toks = fold_emit / fold_wall

    # vanilla greedy reference (correctness + tok/s)
    t0 = time.time(); g_emit = 0; g_out = []
    for p in sel:
        g = vanilla_greedy(base, processor, p["question"], images_dir / p["image"],
                           args.max_new_tokens, eos_id)
        g_emit += len(g); g_out.append(g)
    g_wall = time.time() - t0
    g_toks = g_emit / g_wall

    # correctness: folded vs vanilla greedy common prefix
    prefix_fracs = []; full_match = 0
    for fo, go in zip(fold_out, g_out):
        m = min(len(fo), len(go))
        cp = next((j for j in range(m) if fo[j] != go[j]), m)
        prefix_fracs.append(cp / max(1, m))
        if fo[:m] == go[:m]:
            full_match += 1

    print(f"  baseline gather : tok/s={base_toks:.2f}  σ={statistics.mean(base_sig):.3f}")
    print(f"  folded          : tok/s={fold_toks:.2f}  σ={statistics.mean(fold_sig):.3f}")
    print(f"  vanilla greedy  : tok/s={g_toks:.2f}")
    print(f"  fold/baseline tok/s = {fold_toks / base_toks:.3f}   "
          f"fold net vs greedy = {fold_toks / g_toks:.3f}   baseline net vs greedy = {base_toks / g_toks:.3f}")
    print(f"  correctness (folded vs vanilla greedy): {full_match}/{n} full-prefix match, "
          f"mean common-prefix frac = {statistics.mean(prefix_fracs):.3f}")
    adopt = fold_toks > base_toks
    print(f"\n  ADOPT FOLD? {'YES ✓ (folded faster)' if adopt else 'NO ✗ (not faster)'}  "
          f"[gate: tok/s(folded) > tok/s(baseline)]")
    out = {
        "fanout": args.fanout, "max_nodes": args.max_nodes, "n_prompts": n,
        "baseline_tok_s": base_toks, "baseline_sigma": statistics.mean(base_sig),
        "folded_tok_s": fold_toks, "folded_sigma": statistics.mean(fold_sig),
        "greedy_tok_s": g_toks,
        "fold_over_baseline": fold_toks / base_toks,
        "fold_net_vs_greedy": fold_toks / g_toks, "baseline_net_vs_greedy": base_toks / g_toks,
        "correctness_full_match": full_match, "mean_common_prefix_frac": statistics.mean(prefix_fracs),
        "adopt_fold": adopt,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[save] -> {args.out}")
    return 0


def _regression(base, head, processor, prompts, images_dir, eos_id, args) -> int:
    """BLOCKING gate: fanout=[1,1,1] tree ≡ chain emitted tokens, byte-for-byte."""
    from scripts.eval_acceptance import run_one_prompt as chain_run  # validated chain

    print(f"\n=== REGRESSION: tree(fanout=[1,1,1], reorg={args.reorg}) ≡ chain ===")
    all_match = True
    tree_sigmas, chain_sigmas = [], []
    for i, p in enumerate(prompts):
        chain_r = chain_run(base, head, processor, p, images_dir, args.max_new_tokens, eos_id)
        tree_r = run_one_prompt_tree(base, head, processor, p, images_dir, args.max_new_tokens,
                                     eos_id, [1, 1, 1], max(args.max_nodes, 3),
                                     depth1_floor=False, reorg=args.reorg)
        ct, tt = chain_r["emitted_tokens"], tree_r["emitted_tokens"]
        match = ct == tt
        all_match &= match
        tree_sigmas.append(tree_r["sigma"]); chain_sigmas.append(chain_r["sigma"])
        # first divergence
        div = next((j for j in range(min(len(ct), len(tt))) if ct[j] != tt[j]), None)
        print(f"[reg {i+1}/{len(prompts)}] {p['image']}  "
              f"chain_emit={len(ct)} tree_emit={len(tt)}  "
              f"σ_chain={chain_r['sigma']:.3f} σ_tree={tree_r['sigma']:.3f}  "
              f"MATCH={match}" + ("" if match else f"  first_diverge@{div} (len_eq={len(ct)==len(tt)})"))
    print(f"\n  ALL BYTE-IDENTICAL: {'YES ✓' if all_match else 'NO ✗'}")
    print(f"  mean σ_chain = {statistics.mean(chain_sigmas):.3f}  "
          f"mean σ_tree(width-1) = {statistics.mean(tree_sigmas):.3f}  (expect ≈ 1.95)")
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
