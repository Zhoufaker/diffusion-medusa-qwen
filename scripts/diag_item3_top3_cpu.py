#!/usr/bin/env python3
"""CPU-only: lm_head.float() @ dumped h → fp32 top-3; finish item3 table."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from decode.common import load_base, mask_phantom_  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="/scratch/li96/mz9869/medusa_outputs/"
                    "linked_medusa_c1_eagle/diag_triad/item3_fp32_audit.json")
    ap.add_argument("--out-dir", default="/scratch/li96/mz9869/medusa_outputs/"
                    "linked_medusa_c1_eagle/diag_triad")
    ap.add_argument("--skip-prompt-id", default="000000055049",
                    help="Already-passed row; keep prior top3 if present")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    audit = json.load(open(args.audit))
    print("[3a-cpu] loading base for lm_head only (will move weight to CPU)")
    base, _ = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    lm_w = base.lm_head.weight.detach().float().cpu()
    del base
    torch.cuda.empty_cache()

    # preserve existing 055049 row if present
    outp = out_dir / "item3_fp32_top3.json"
    prev = {}
    if outp.exists():
        for r in json.load(open(outp)):
            prev[r["prompt_id"]] = r

    out_rows = []
    for r in audit:
        pid = r["prompt_id"]
        pos = r["pos"]
        h_path = out_dir / "item3_h_fp16" / f"{pid}_pos{pos}.pt"
        if not h_path.exists():
            if pid == args.skip_prompt_id and pid in prev:
                out_rows.append(prev[pid])
                print(f"  keep prior {pid}")
                continue
            print(f"  MISSING h {h_path}")
            continue
        blob = torch.load(h_path, map_location="cpu", weights_only=True)
        h = blob["h"].float()
        logits = mask_phantom_(torch.nn.functional.linear(h, lm_w))
        top3 = torch.topk(logits, 3)
        ids = [int(x) for x in top3.indices.tolist()]
        vals = [float(x) for x in top3.values.tolist()]
        tree = r["tree_emitted_tok"]
        rec = {
            **{k: r[k] for k in ("prompt_id", "band", "pos", "tree_emitted_tok",
                                 "greedy_emitted_tok", "fp32_argmax", "fp32_top2_gap",
                                 "fp16_causal_eq_fp32")},
            "fp32_top3_ids": ids,
            "fp32_top3_logits": vals,
            "tree_in_top2": tree in ids[:2],
            "tree_in_top3": tree in ids[:3],
            "method": "process-isolated h_dump_fp16 + lm_head.float() @ h (CPU)",
        }
        out_rows.append(rec)
        print(f"  {pid} pos={pos} tree={tree} top3={ids} in_top2={rec['tree_in_top2']} "
              f"eq={r['fp16_causal_eq_fp32']}")

    json.dump(out_rows, open(outp, "w"), indent=2)
    eq = [r for r in out_rows if r["fp16_causal_eq_fp32"]]
    fail = [r for r in eq if not r["tree_in_top2"]]
    print(f"[3a-cpu] n={len(out_rows)}/10  16=32=Y={len(eq)}  tree∉top-2={len(fail)}")
    for r in fail:
        print(f"  FAIL {r['prompt_id']} tree={r['tree_emitted_tok']} top3={r['fp32_top3_ids']}")
    return 2 if fail else (0 if len(out_rows) >= 10 else 1)


if __name__ == "__main__":
    sys.exit(main())
