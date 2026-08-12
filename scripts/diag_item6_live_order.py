#!/usr/bin/env python3
"""3b live order confrontation: static vs dynamic per-round ordered flat tables.

For σ-drift prompts: run both builders live (gate#2), dump per-round ordered
nodes + accepted + mask/pos sha. Find first accepted-seq diverge round r*;
compare ordered flats. Harness-only — decode/ untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from decode.common import (  # noqa: E402
    argmax_masked,
    cfg_attr,
    continuation_base,
    filter_prompts,
    load_base,
    load_head,
    make_image_inputs,
)
from decode.tree import (  # noqa: E402
    accept,
    build_mask_and_positions,
    build_tree_folded,
    build_tree_folded_dynamic,
    reorg_kv_gather,
    tree_tokens,
)


def _sha_tensor(t: torch.Tensor) -> str:
    b = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def _flat_rows(nodes) -> List[dict]:
    return [
        {
            "flat_idx": int(n.flat_idx),
            "token": int(n.token),
            "depth": int(n.depth),
            "parent_flat_idx": int(n.parent),
            "cum": f"{float(n.cum_logprob):.12g}",
            "logprob": f"{float(n.logprob):.12g}",
        }
        for n in nodes  # already flat order 0..N-1
    ]


def _path_set(nodes) -> set:
    by = {n.flat_idx: n for n in nodes}

    def path(n):
        toks = []
        cur = n
        while True:
            toks.append(int(cur.token))
            if cur.parent == -1:
                break
            cur = by[cur.parent]
        return tuple(reversed(toks))

    return {path(n) for n in nodes}


@torch.no_grad()
def run_live_dump(base, head, processor, prompt, images_dir, fanout, max_nodes,
                  max_new, eos_id, builder: str, device="cuda:0") -> Dict:
    _build = build_tree_folded if builder == "static" else build_tree_folded_dynamic
    inputs = make_image_inputs(processor, prompt["question"],
                               images_dir / prompt["image"], device)
    out = base(**inputs, use_cache=True, output_hidden_states=True)
    past_kv = out.past_key_values
    P = past_kv.get_seq_length()
    h_anchor = out.hidden_states[-1][0, -1, :].clone()
    known_next = argmax_masked(out.logits[0, -1, :])
    del out
    torch.cuda.empty_cache()

    n_active = len(fanout)
    while n_active > 1 and fanout[n_active - 1] == 0:
        n_active -= 1
    fan_a = list(fanout[:n_active])

    emitted: List[int] = []
    rounds = []
    rd = 0
    while len(emitted) < max_new:
        rd += 1
        cont_base = continuation_base(base, P)
        cond = base.get_input_embeddings()(
            torch.tensor([[known_next]], device=device, dtype=torch.long)
        )
        all_logits = head(
            h_anchor.view(1, 1, -1).half(), max_heads=n_active,
            skip_head0_lm_head=True, cond_embed=cond,
        )
        nodes = _build(all_logits, int(known_next), fan_a, max_nodes, True)
        del all_logits, cond
        mask, pos = build_mask_and_positions(nodes, P, cont_base, base.dtype, device)
        toks = tree_tokens(nodes, device)
        rec = {
            "round": rd - 1,
            "known_next": int(known_next),
            "n_nodes": len(nodes),
            "flat": _flat_rows(nodes),
            "path_set": [list(p) for p in sorted(_path_set(nodes))],
            "position_ids_sha": _sha_tensor(pos),
            "mask_sha": _sha_tensor(mask),
            "emitted_before": list(emitted),
        }
        v_out = base(
            input_ids=toks, attention_mask=mask, past_key_values=past_kv,
            position_ids=pos, use_cache=True, output_hidden_states=True,
        )
        accepted, alen, bonus, _ = accept(nodes, v_out.logits[0], known_next)
        accepted_tokens = [nodes[i].token for i in accepted]
        rec["accepted_flat_idx"] = list(accepted)
        rec["accepted_tokens"] = accepted_tokens
        rec["accept_len"] = int(alen)
        rec["bonus"] = int(bonus)
        rounds.append(rec)

        if not accepted:
            break
        if eos_id in accepted_tokens:
            cut = accepted_tokens.index(eos_id) + 1
            emitted.extend(accepted_tokens[:cut])
            break
        emitted.extend(accepted_tokens)
        last = accepted[-1]
        h_anchor = v_out.hidden_states[-1][0, last, :].clone()
        reorg_kv_gather(v_out.past_key_values, P, accepted, device)
        past_kv = v_out.past_key_values
        P = past_kv.get_seq_length()
        known_next = bonus
        del v_out, nodes, mask, pos, toks
        torch.cuda.empty_cache()
        if bonus == eos_id:
            break
    del past_kv
    torch.cuda.empty_cache()
    return {
        "builder": builder,
        "prompt_id": str(prompt["id"]),
        "emitted_tokens": emitted,
        "rounds": rounds,
    }


def compare_prompt(static: Dict, dynamic: Dict) -> Dict:
    """Find r* = first round where accepted_tokens differ; compare flats."""
    ns, nd = len(static["rounds"]), len(dynamic["rounds"])
    r_star = None
    for r in range(min(ns, nd)):
        if static["rounds"][r]["accepted_tokens"] != dynamic["rounds"][r]["accepted_tokens"]:
            r_star = r
            break
    # also check emitted prefix length mismatch if one side ended early
    if r_star is None and ns != nd:
        r_star = min(ns, nd)  # diverge by length after last common

    verdict = None
    detail = None
    if r_star is None:
        # entire accepted traces match per-round — check full emitted
        same_emit = static["emitted_tokens"] == dynamic["emitted_tokens"]
        verdict = "no_accepted_diverge"
        detail = {"emitted_identical": same_emit,
                  "n_rounds_s": ns, "n_rounds_d": nd}
    else:
        if r_star >= ns or r_star >= nd:
            verdict = "length_mismatch"
            detail = {"r_star": r_star, "n_s": ns, "n_d": nd}
        else:
            rs, rd = static["rounds"][r_star], dynamic["rounds"][r_star]
            set_s = {tuple(p) for p in rs["path_set"]}
            set_d = {tuple(p) for p in rd["path_set"]}
            flat_s = rs["flat"]
            flat_d = rd["flat"]
            # ordered equality
            order_eq = flat_s == flat_d
            set_eq = set_s == set_d
            # parent/token sequence ignoring flat_idx numbering: compare by path
            if set_eq and not order_eq:
                verdict = "set_eq_order_ne_reflatten"
            elif set_eq and order_eq:
                verdict = "order_eq_upgrade_stop"  # contradicts same-compute
            elif not set_eq:
                verdict = "set_ne_dump_fidelity"
            else:
                verdict = "unknown"
            detail = {
                "r_star": r_star,
                "accepted_static": rs["accepted_tokens"],
                "accepted_dynamic": rd["accepted_tokens"],
                "known_next_s": rs["known_next"],
                "known_next_d": rd["known_next"],
                "n_s": rs["n_nodes"],
                "n_d": rd["n_nodes"],
                "set_equal": set_eq,
                "order_equal": order_eq,
                "only_static_paths": [list(p) for p in sorted(set_s - set_d)],
                "only_dynamic_paths": [list(p) for p in sorted(set_d - set_s)],
                "flat_static": flat_s,
                "flat_dynamic": flat_d,
                "position_ids_sha_s": rs["position_ids_sha"],
                "position_ids_sha_d": rd["position_ids_sha"],
                "mask_sha_s": rs["mask_sha"],
                "mask_sha_d": rd["mask_sha"],
                "pos_sha_equal": rs["position_ids_sha"] == rd["position_ids_sha"],
                "mask_sha_equal": rs["mask_sha"] == rd["mask_sha"],
            }
    return {
        "prompt_id": static["prompt_id"],
        "verdict": verdict,
        "detail": detail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="/g/data/li96/mz9869/data/llava_subset_2k.json")
    ap.add_argument("--images-dir", default="/g/data/li96/mz9869/data/coco_subset")
    ap.add_argument("--indices", type=int, nargs="+", default=[10, 58, 78, 85])
    ap.add_argument("--fanout", type=int, nargs="+", default=[1, 6, 4, 2, 1])
    ap.add_argument("--max-nodes", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_all = filter_prompts(args.manifest, 80, 42, ordered=False)[:100]
    prompts = [(i, prompts_all[i]) for i in args.indices]

    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    head = load_head(args.ckpt, cfg_attr(base.config, "hidden_size"),
                     cfg_attr(base.config, "vocab_size"))
    eos = processor.tokenizer.eos_token_id
    images = Path(args.images_dir)

    reports = []
    for i, p in prompts:
        print(f"\n=== drift i={i} id={p['id']} ===")
        print("[live] static...")
        st = run_live_dump(base, head, processor, p, images, args.fanout,
                           args.max_nodes, args.max_new_tokens, eos, "static")
        json.dump(st, open(out_dir / f"live_static_{p['id']}.json", "w"))
        print(f"  static rounds={len(st['rounds'])} emit={len(st['emitted_tokens'])}")
        torch.cuda.empty_cache()
        print("[live] dynamic...")
        dy = run_live_dump(base, head, processor, p, images, args.fanout,
                           args.max_nodes, args.max_new_tokens, eos, "dynamic")
        json.dump(dy, open(out_dir / f"live_dynamic_{p['id']}.json", "w"))
        print(f"  dynamic rounds={len(dy['rounds'])} emit={len(dy['emitted_tokens'])}")
        torch.cuda.empty_cache()

        cmp = compare_prompt(st, dy)
        reports.append(cmp)
        print(f"[verdict] {cmp['verdict']}")
        d = cmp["detail"] or {}
        if cmp["verdict"] == "set_eq_order_ne_reflatten":
            print(f"  r*={d['r_star']} set_eq order_ne → reflatten/parent-remap suspect")
            print(f"  pos_sha eq={d['pos_sha_equal']} mask_sha eq={d['mask_sha_equal']}")
            print("  --- flat static (r*) ---")
            for row in d["flat_static"]:
                print(f"    {row}")
            print("  --- flat dynamic (r*) ---")
            for row in d["flat_dynamic"]:
                print(f"    {row}")
        elif cmp["verdict"] == "order_eq_upgrade_stop":
            print(f"  r*={d['r_star']} ORDER ALSO EQUAL — upgrade stop")
            print(f"  accepted_s={d['accepted_static']}")
            print(f"  accepted_d={d['accepted_dynamic']}")
        elif cmp["verdict"] == "set_ne_dump_fidelity":
            print(f"  r*={d['r_star']} SET UNEQUAL")
            print(f"  only_s={d['only_static_paths'][:5]}")
            print(f"  only_d={d['only_dynamic_paths'][:5]}")
        else:
            print(f"  detail={json.dumps(d)[:500]}")

    out = {"reports": reports}
    json.dump(out, open(out_dir / "live_order_verdicts.json", "w"), indent=2)
    print(f"\n[wrote] {out_dir / 'live_order_verdicts.json'}")

    # If any reflatten verdict, print reflatten code pointers and STOP
    if any(r["verdict"] == "set_eq_order_ne_reflatten" for r in reports):
        print("\n=== STOP: reflatten/parent-remap inconsistency suspected ===")
        print("Static reflatten: decode/tree.py build_tree_folded ~L180-187")
        print("Dynamic reflatten: decode/tree.py build_tree_folded_dynamic ~L277-292")
        print("Awaiting advisor fix direction (dynamic must byte-match static flat layout).")
    if any(r["verdict"] == "order_eq_upgrade_stop" for r in reports):
        print("\n=== UPGRADE STOP: order identical but accepted diverged ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
