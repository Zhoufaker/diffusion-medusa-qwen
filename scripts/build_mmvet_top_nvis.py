#!/usr/bin/env python3
"""Build mini-manifest of top-K MM-Vet prompts by n_vis under max_pixels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from decode.common import filter_prompts, load_base, make_image_inputs  # noqa: E402
from scripts.diag_prefill_mem import apply_max_pixels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pixels", type=int, default=501760)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="/scratch/li96/mz9869/ood_eval/mmvet/manifest_mmvet_218.json")
    ap.add_argument("--images", default="/scratch/li96/mz9869/ood_eval/mmvet/mm-vet/images")
    args = ap.parse_args()

    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    apply_max_pixels(processor, args.max_pixels)
    prompts = filter_prompts(args.manifest, 0, 42, ordered=True)
    images = Path(args.images)
    rows = []
    for p in prompts:
        # resolve image: manifest may say .png but file is .jpg
        img = images / p["image"]
        if not img.exists():
            stem = Path(p["image"]).stem
            cands = list(images.glob(stem + ".*"))
            if not cands:
                print(f"MISSING {p['id']} {p['image']}")
                continue
            img = cands[0]
            p = {**p, "image": img.name}
        inputs = make_image_inputs(processor, p["question"], img, "cuda:0")
        thw = inputs.get("image_grid_thw")
        n_vis = int(thw.prod().item() // 4) if thw is not None else -1
        seq = int(inputs["input_ids"].shape[-1])
        rows.append({**p, "n_vision_tokens_est": n_vis, "seq_len": seq,
                     "image_grid_thw": thw.tolist() if thw is not None else None})
        del inputs
    rows.sort(key=lambda r: r["n_vision_tokens_est"], reverse=True)
    top = rows[: args.top_k]
    # Emit LLaVA-style conversations so filter_prompts accepts the mini-manifest
    out_prompts = []
    for r in top:
        out_prompts.append({
            "id": r["id"],
            "image": r["image"],
            "conversations": [
                {"from": "human", "value": r["question"] + "\n<image>"},
                {"from": "gpt", "value": r["answer"]},
            ],
            "n_vision_tokens_est": r["n_vision_tokens_est"],
            "seq_len": r["seq_len"],
            "image_grid_thw": r["image_grid_thw"],
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_prompts, open(args.out, "w"), indent=2)
    print(f"[top{args.top_k}] wrote {args.out}")
    for r in top:
        print(f"  {r['id']} image={r['image']} n_vis={r['n_vision_tokens_est']} "
              f"seq={r['seq_len']} thw={r['image_grid_thw']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
