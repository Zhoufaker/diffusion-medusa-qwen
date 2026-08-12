#!/usr/bin/env python3
"""O0 item3 fp32 audit + dynamic gate#2 tree-diff. Memory-safe on V100 32GB."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decode.common import (  # noqa: E402
    argmax_masked,
    cfg_attr,
    continuation_base,
    filter_prompts,
    load_base,
    load_head,
    make_image_inputs,
    mask_phantom_,
    vanilla_greedy,
)
from decode.tree import (  # noqa: E402
    accept,
    build_mask_and_positions,
    build_tree_folded,
    build_tree_folded_dynamic,
    per_depth_widths,
    reorg_kv_gather,
    tree_tokens,
)


def _free():
    torch.cuda.empty_cache()


def hist_match(spec: List[int], greedy: List[int]) -> Tuple[bool, int, int]:
    m = min(len(spec), len(greedy))
    cp = next((j for j in range(m) if spec[j] != greedy[j]), m)
    return spec[:m] == greedy[:m], cp, m


@torch.no_grad()
def replay_logits_and_hidden(base, processor, question, image_path, prefix):
    """Causal replay in model dtype; returns (logits_fp16, last_hidden_fp16)."""
    device = "cuda:0"
    inputs = make_image_inputs(processor, question, image_path, device)
    out = base(**inputs, use_cache=True, output_hidden_states=True)
    past = out.past_key_values
    logits = mask_phantom_(out.logits[0, -1, :])
    h = out.hidden_states[-1][0, -1, :]
    for tok in prefix:
        out = base(
            input_ids=torch.tensor([[tok]], device=device),
            past_key_values=past, use_cache=True, output_hidden_states=True,
        )
        past = out.past_key_values
        logits = mask_phantom_(out.logits[0, -1, :])
        h = out.hidden_states[-1][0, -1, :]
    # detach clones so KV can be freed
    logits = logits.detach().clone()
    h = h.detach().clone()
    del out, past, inputs
    _free()
    return logits, h


def node_records(nodes) -> Dict[Tuple[int, ...], dict]:
    by = {n.flat_idx: n for n in nodes}

    def path_of(n):
        toks = []
        cur = n
        while True:
            toks.append(int(cur.token))
            if cur.parent == -1:
                break
            cur = by[cur.parent]
        return tuple(reversed(toks))

    children: Dict[int, List[int]] = {}
    for n in nodes:
        children.setdefault(n.parent, []).append(n.flat_idx)
    slot = {}
    for idxs in children.values():
        for b, fi in enumerate(idxs):
            slot[fi] = b

    out = {}
    for n in nodes:
        p = path_of(n)
        cum = float(n.cum_logprob)
        out[p] = {
            "token": int(n.token),
            "depth": int(n.depth),
            "path": list(p),
            "slot_b": int(slot[n.flat_idx]),
            "parent_path": list(p[:-1]) if len(p) > 1 else [],
            "cum": cum,
            "cum_repr": f"{cum:.17g}",
            "cum_hex": cum.hex(),
            "logprob": float(n.logprob),
            "logprob_hex": float(n.logprob).hex(),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt", default="/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/ckpt_best.pt")
    ap.add_argument("--manifest300", default="/scratch/li96/mz9869/eval_manifests/manifest_300.json")
    ap.add_argument("--manifest-llava", default="/g/data/li96/mz9869/data/llava_subset_2k.json")
    ap.add_argument("--images", default="/g/data/li96/mz9869/data/coco_subset")
    ap.add_argument("--o0-div-json",
                    default="/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/bridge_300/c1_d3.o0_divergences.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=["item3", "item6", "all"], default="all")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    eos_id = processor.tokenizer.eos_token_id
    images_dir = Path(args.images)

    divs = json.load(open(args.o0_div_json))
    man = json.load(open(args.manifest300))
    id_to_idx = {x["id"]: i for i, x in enumerate(man)}

    # ---- offline item2 echo ----
    old_mid = [r for r in divs
               if id_to_idx[r["prompt_id"]] < 100
               and r["pos"] < min(r["spec_len"], r["greedy_len"])]
    print(f"[item2-offline] hist match {100 - len(old_mid)}/100 mid_fail={len(old_mid)}")

    if args.only in ("item3", "all"):
        mid = [r for r in divs
               if r["pos"] < min(r["spec_len"], r["greedy_len"])
               and r.get("top2_logit_gap") is not None]
        rng = random.Random(args.seed)
        old_m = [r for r in mid if id_to_idx[r["prompt_id"]] < 100]
        new_m = [r for r in mid if id_to_idx[r["prompt_id"]] >= 100]
        sample = rng.sample(old_m, min(5, len(old_m))) + rng.sample(new_m, min(5, len(new_m)))
        prompts300 = filter_prompts(args.manifest300, 80, 42, ordered=True)
        by_id = {p["id"]: p for p in prompts300}
        print(f"[item3] fp32 audit n={len(sample)} (no head loaded; fp32=lm_head.float on fp16 h)")

        # lm_head fp32 copy for logits recompute (full model.float() OOMs on V100)
        lm_w = base.lm_head.weight.detach().float().cpu()
        fp32_rows = []
        for r in sample:
            p = by_id[r["prompt_id"]]
            pos = int(r["pos"])
            _free()
            g = vanilla_greedy(base, processor, p["question"], images_dir / p["image"], 150, eos_id)
            _free()
            prefix = g[:pos]
            # tree emitted token taken from O0 dump (live tree choice); greedy from this run
            tree_tok = r["spec_tok"]
            greedy_tok = g[pos] if pos < len(g) else None
            # sanity: dump greedy_tok should match re-run if deterministic
            dump_g = r["greedy_tok"]

            lg16, h16 = replay_logits_and_hidden(
                base, processor, p["question"], images_dir / p["image"], prefix,
            )
            am16 = argmax_masked(lg16)
            top2_16 = torch.topk(lg16.float(), 2).values
            gap16 = float(top2_16[0] - top2_16[1])

            # fp32 logits = h.float() @ W.float().T  (lm_head)
            logits32 = torch.nn.functional.linear(h16.float().cpu(), lm_w)
            # apply same phantom mask on CPU vocab slice — reuse GPU mask via copy
            logits32_g = logits32.to(lg16.device)
            logits32_g = mask_phantom_(logits32_g)
            am32 = argmax_masked(logits32_g)
            top2_32 = torch.topk(logits32_g.float(), 2).values
            gap32 = float(top2_32[0] - top2_32[1])
            del lg16, h16, logits32, logits32_g
            _free()

            row = {
                "prompt_id": r["prompt_id"],
                "band": "old100" if id_to_idx[r["prompt_id"]] < 100 else "new200",
                "pos": pos,
                "fp32_method": "lm_head.float() @ hidden.float() after fp16 backbone replay",
                "fp32_top2_gap": gap32,
                "fp16_causal_top2_gap": gap16,
                "fp16_o0_recorded_gap": r.get("top2_logit_gap"),
                "fp32_argmax": am32,
                "fp16_causal_argmax": am16,
                "fp16_causal_eq_fp32": am16 == am32,
                "tree_emitted_tok": tree_tok,
                "greedy_emitted_tok": greedy_tok,
                "o0_dump_greedy_tok": dump_g,
                "greedy_rerun_eq_o0_dump": greedy_tok == dump_g,
                "tree_emitted_eq_fp32": tree_tok == am32,
                "greedy_emitted_eq_fp32": greedy_tok == am32,
                "tree_emitted_eq_fp16_causal": tree_tok == am16,
                "greedy_emitted_eq_fp16_causal": greedy_tok == am16,
            }
            fp32_rows.append(row)
            print(
                f"  {row['prompt_id']} [{row['band']}] pos={pos} "
                f"gap32={gap32:.6f} am32={am32} am16={am16} eq={am16==am32} "
                f"tree={tree_tok} greedy={greedy_tok} "
                f"tree=fp32?{row['tree_emitted_eq_fp32']} greedy=fp32?{row['greedy_emitted_eq_fp32']}"
            )
        json.dump(fp32_rows, open(out_dir / "item3_fp32_audit.json", "w"), indent=2)
        print(f"[item3] wrote {out_dir / 'item3_fp32_audit.json'}")

    if args.only in ("item6", "all"):
        print("[item6] loading head for tree diff")
        head = load_head(args.ckpt, cfg_attr(base.config, "hidden_size"),
                         cfg_attr(base.config, "vocab_size"))
        drift_idx = [10, 58, 78, 85]
        nbar_idx = [13, 61, 88]
        llava = filter_prompts(args.manifest_llava, 80, 42, ordered=False)[:100]
        fanout = [1, 6, 4, 2, 1]
        max_nodes = 24
        device = "cuda:0"

        def scan_prompt(p, max_rounds=80, stop_on_set_diff=True, collect_n_diff=5):
            _free()
            inputs = make_image_inputs(processor, p["question"], images_dir / p["image"], device)
            out = base(**inputs, use_cache=True, output_hidden_states=True)
            past_kv = out.past_key_values
            P = past_kv.get_seq_length()
            h_anchor = out.hidden_states[-1][0, -1, :].clone()
            known_next = argmax_masked(out.logits[0, -1, :])
            del out
            _free()
            n_active = len(fanout)
            while n_active > 1 and fanout[n_active - 1] == 0:
                n_active -= 1
            fan_a = list(fanout[:n_active])
            rounds = []
            n_diff_kept = 0
            for rd in range(max_rounds):
                cond = base.get_input_embeddings()(
                    torch.tensor([[known_next]], device=device, dtype=torch.long)
                )
                all_logits = head(
                    h_anchor.view(1, 1, -1).half(), max_heads=n_active,
                    skip_head0_lm_head=True, cond_embed=cond,
                )
                # Offload logits to CPU before dual build + verify (diag OOM root cause)
                cpu_logits = [
                    x.detach().float().cpu() if x is not None else None for x in all_logits
                ]
                del all_logits, cond
                _free()
                ns = build_tree_folded(cpu_logits, int(known_next), fan_a, max_nodes, True)
                nd = build_tree_folded_dynamic(cpu_logits, int(known_next), fan_a, max_nodes, True)
                ss, sd = node_records(ns), node_records(nd)
                only_s = [ss[k] for k in ss.keys() - sd.keys()]
                only_d = [sd[k] for k in sd.keys() - ss.keys()]
                cum_diff = []
                for k in ss.keys() & sd.keys():
                    if ss[k]["cum"] != sd[k]["cum"]:
                        cum_diff.append({
                            "path": list(k),
                            "static_cum": ss[k]["cum_repr"],
                            "dynamic_cum": sd[k]["cum_repr"],
                            "static_hex": ss[k]["cum_hex"],
                            "dynamic_hex": sd[k]["cum_hex"],
                        })
                rec = {
                    "round": rd,
                    "known_next": int(known_next),
                    "n_static": len(ns),
                    "n_dynamic": len(nd),
                    "widths_static": per_depth_widths(ns, n_active),
                    "widths_dynamic": per_depth_widths(nd, n_active),
                    "only_static": only_s,
                    "only_dynamic": only_d,
                    "cum_diff_same_path": cum_diff,
                    "set_equal": (not only_s) and (not only_d) and (not cum_diff),
                }
                interesting = (not rec["set_equal"]) or (len(ns) != len(nd))
                if interesting:
                    rounds.append(rec)
                    n_diff_kept += 1
                    if stop_on_set_diff and not rec["set_equal"]:
                        break
                    if (not stop_on_set_diff) and n_diff_kept >= collect_n_diff:
                        break
                del nd, cpu_logits, ss, sd
                _free()
                cont_base = continuation_base(base, P)
                mask, pos = build_mask_and_positions(ns, P, cont_base, base.dtype, device)
                toks = tree_tokens(ns, device)
                v_out = base(
                    input_ids=toks, attention_mask=mask, past_key_values=past_kv,
                    position_ids=pos, use_cache=True, output_hidden_states=True,
                )
                accepted, _alen, bonus, _ = accept(ns, v_out.logits[0], known_next)
                if not accepted:
                    break
                last = accepted[-1]
                h_anchor = v_out.hidden_states[-1][0, last, :].clone()
                reorg_kv_gather(v_out.past_key_values, P, accepted, device)
                past_kv = v_out.past_key_values
                P = past_kv.get_seq_length()
                known_next = bonus
                del v_out, ns, mask, pos, toks
                _free()
                if bonus == eos_id:
                    break
            del past_kv
            _free()
            return rounds

        sigma_reports = []
        for i in drift_idx:
            p = llava[i]
            print(f"[item6] σ-drift i={i} id={p['id']}")
            rounds = scan_prompt(p, stop_on_set_diff=True)
            first = rounds[0] if rounds else None
            sigma_reports.append({"prompt_index": i, "prompt_id": p["id"], "first_diff": first})
            if first:
                print(
                    f"  round={first['round']} n_s={first['n_static']} n_d={first['n_dynamic']} "
                    f"only_s={len(first['only_static'])} only_d={len(first['only_dynamic'])} "
                    f"cum_diff={len(first['cum_diff_same_path'])}"
                )
            else:
                print("  no set-diff in scanned rounds")

        nbar_reports = []
        for i in nbar_idx:
            p = llava[i]
            print(f"[item6] N̄-sample i={i} id={p['id']}")
            rounds = scan_prompt(p, stop_on_set_diff=False, collect_n_diff=5)
            nbar_reports.append({
                "prompt_index": i, "prompt_id": p["id"],
                "diff_rounds": rounds, "n_diff_rounds": len(rounds),
            })
            print(f"  kept_diff_rounds={len(rounds)}")

        json.dump(
            {"sigma_drift": sigma_reports, "nbar_samples": nbar_reports},
            open(out_dir / "item6_tree_diff.json", "w"), indent=2,
        )
        print(f"[item6] wrote {out_dir / 'item6_tree_diff.json'}")

    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
