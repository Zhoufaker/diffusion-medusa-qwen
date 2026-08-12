#!/usr/bin/env python3
"""Prefill-only VRAM scan for MM-Vet 218 under max_pixels (V100-safe: no head in main pass)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decode.common import (  # noqa: E402
    apply_max_pixels,
    argmax_masked,
    cfg_attr,
    continuation_base,
    filter_prompts,
    load_base,
    load_head,
    make_image_inputs,
)
from decode.tree import (  # noqa: E402
    build_mask_and_positions,
    build_tree_folded,
    tree_tokens,
)

# re-export for older imports: from scripts.diag_prefill_mem import apply_max_pixels
__all__ = ["apply_max_pixels", "main"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pixels", type=int, required=True)
    ap.add_argument("--manifest", default="/scratch/li96/mz9869/ood_eval/mmvet/manifest_mmvet_218.json")
    ap.add_argument("--images", default="/scratch/li96/mz9869/ood_eval/mmvet/mm-vet/images")
    ap.add_argument("--ckpt", default="/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/ckpt_best.pt")
    ap.add_argument("--fanout", type=int, nargs="+", default=[1, 6, 4, 3, 2])
    ap.add_argument("--max-nodes", type=int, default=32)
    ap.add_argument("--tree-probe-top", type=int, default=3,
                    help="After prefill scan, load head and tree-probe this many highest-seq prompts")
    args = ap.parse_args()

    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    apply_max_pixels(processor, args.max_pixels)
    print(f"[cap] max_pixels={args.max_pixels}  ip.size={processor.image_processor.size}")
    print(f"[mem] after base load: {torch.cuda.memory_allocated()/(1024**3):.2f} GiB")

    prompts = filter_prompts(args.manifest, min_ref_words=0, seed=42, ordered=True)
    assert len(prompts) == 218, len(prompts)
    images_dir = Path(args.images)
    device = "cuda:0"
    rows = []

    for i, p in enumerate(prompts):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem0 = torch.cuda.memory_allocated()
        inputs = make_image_inputs(processor, p["question"], images_dir / p["image"], device)
        seq_len = int(inputs["input_ids"].shape[-1])
        out = base(**inputs, use_cache=True)
        peak_prefill = torch.cuda.max_memory_allocated()
        del out, inputs
        torch.cuda.empty_cache()
        row = {
            "i": i,
            "id": p["id"],
            "image": p["image"],
            "seq_len": seq_len,
            "peak_prefill_bytes": int(peak_prefill),
            "peak_prefill_gib": peak_prefill / (1024 ** 3),
            "mem0_gib": mem0 / (1024 ** 3),
        }
        rows.append(row)
        if (i + 1) % 20 == 0 or i < 5:
            print(f"[{i+1}/218] {p['id']} seq={seq_len} peak={row['peak_prefill_gib']:.2f} GiB")

    rows_sorted = sorted(rows, key=lambda r: (-r["peak_prefill_bytes"], -r["seq_len"]))
    top = rows_sorted[: max(3, args.tree_probe_top)]
    print("--- TOP3 prefill peaks (base only, no head) ---")
    for r in top[:3]:
        print(f"  {r['id']} {r['image']} seq={r['seq_len']} peak={r['peak_prefill_gib']:.3f} GiB")

    # Tree probe on highest-seq prompts with head loaded (production-like peak)
    tree_rows = []
    if args.tree_probe_top > 0:  # 0 = base-only scan (no head)
        print(f"[tree-probe] loading head; probing top-{args.tree_probe_top} by seq_len")
        head = load_head(args.ckpt, cfg_attr(base.config, "hidden_size"),
                         cfg_attr(base.config, "vocab_size"))
        print(f"[mem] after head load: {torch.cuda.memory_allocated()/(1024**3):.2f} GiB")
        by_seq = sorted(rows, key=lambda r: -r["seq_len"])[: args.tree_probe_top]
        # also include absolute prefill-peak tops
        ids = []
        for r in by_seq + top[:3]:
            if r["id"] not in ids:
                ids.append(r["id"])
        id_to_p = {p["id"]: p for p in prompts}
        id_to_row = {r["id"]: r for r in rows}
        for pid in ids[: args.tree_probe_top]:
            p = id_to_p[pid]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            inputs = make_image_inputs(processor, p["question"], images_dir / p["image"], device)
            out = base(**inputs, use_cache=True, output_hidden_states=True)
            past_kv = out.past_key_values
            P = past_kv.get_seq_length()
            h_anchor = out.hidden_states[-1][0, -1, :].clone()
            known_next = argmax_masked(out.logits[0, -1, :])
            del out
            fan = list(args.fanout)
            n_active = len(fan)
            while n_active > 1 and fan[n_active - 1] == 0:
                n_active -= 1
            fan = fan[:n_active]
            cond = base.get_input_embeddings()(
                torch.tensor([[known_next]], device=device, dtype=torch.long)
            )
            all_logits = head(
                h_anchor.view(1, 1, -1).half(), max_heads=n_active,
                skip_head0_lm_head=True, cond_embed=cond,
            )
            nodes = build_tree_folded(all_logits, int(known_next), fan, args.max_nodes, True)
            cont = continuation_base(base, P)
            mask, pos = build_mask_and_positions(nodes, P, cont, base.dtype, device)
            toks = tree_tokens(nodes, device)
            _ = base(
                input_ids=toks, attention_mask=mask, past_key_values=past_kv,
                position_ids=pos, use_cache=True,
            )
            peak = torch.cuda.max_memory_allocated()
            tr = {
                "id": pid,
                "image": p["image"],
                "seq_len": id_to_row[pid]["seq_len"],
                "n_nodes": len(nodes),
                "peak_tree_gib": peak / (1024 ** 3),
            }
            tree_rows.append(tr)
            print(f"  tree-probe {pid} seq={tr['seq_len']} N={tr['n_nodes']} "
                  f"peak={tr['peak_tree_gib']:.3f} GiB")
            del past_kv, all_logits, nodes, _, inputs
            torch.cuda.empty_cache()

    budget = 31.73
    cap = budget * 0.90
    max_prefill = rows_sorted[0]["peak_prefill_gib"]
    max_tree = max((t["peak_tree_gib"] for t in tree_rows), default=None)
    summary = {
        "max_pixels": args.max_pixels,
        "fanout": args.fanout,
        "max_nodes": args.max_nodes,
        "n": len(rows),
        "max_prefill_gib_base_only": max_prefill,
        "top3_prefill": top[:3],
        "tree_probe": tree_rows,
        "max_tree_probe_gib": max_tree,
        "budget_gib": budget,
        "budget_cap_gib": cap,
        "fits_10pct_prefill_base_only": max_prefill <= cap,
        "fits_10pct_tree_probe": (None if max_tree is None else max_tree <= cap),
    }
    out = {"summary": summary, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"summary: {json.dumps({k: summary[k] for k in summary if k not in ('top3_prefill','tree_probe')}, indent=2)}")
    print(f"[save] {args.out}")
    if max_tree is not None and not summary["fits_10pct_tree_probe"]:
        print("[WARN] tree-probe peak exceeds 10% headroom — lower max_pixels further")
        return 2
    if not summary["fits_10pct_prefill_base_only"]:
        print("[WARN] base-only prefill peak exceeds 10% headroom")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
