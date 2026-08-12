#!/usr/bin/env python3
"""Item5(b) mem forensics on one MM-Vet image under a given max_pixels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from decode.common import load_base, make_image_inputs  # noqa: E402
from scripts.diag_prefill_mem import apply_max_pixels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pixels", type=int, default=501760)
    ap.add_argument("--image", default="/scratch/li96/mz9869/ood_eval/mmvet/mm-vet/images/v1_82.png")
    ap.add_argument("--question", default="Describe the image.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()
    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    apply_max_pixels(processor, args.max_pixels)
    after_load_alloc = torch.cuda.memory_allocated()
    after_load_rsrv = torch.cuda.memory_reserved()

    from PIL import Image
    im = Image.open(args.image).convert("RGB")
    raw_wh = im.size
    raw_px = raw_wh[0] * raw_wh[1]

    inputs = make_image_inputs(processor, args.question, Path(args.image), "cuda:0")
    seq_len = int(inputs["input_ids"].shape[-1])
    # vision tokens ≈ count of image pad / unique image token id; also pixel_values shape
    pv = inputs.get("pixel_values")
    pv_shape = list(pv.shape) if pv is not None else None
    # Qwen2-VL: grid_thw gives token geometry
    thw = inputs.get("image_grid_thw")
    thw_list = thw.tolist() if thw is not None else None
    n_vision = None
    if thw is not None:
        # tokens after merge: t*h*w / (merge^2) with merge=2 → t*h*w/4
        n_vision = int(thw.prod().item() // 4)

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = base(**inputs, use_cache=True)
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_rsrv = torch.cuda.max_memory_reserved()
    after = torch.cuda.memory_allocated()
    del out
    torch.cuda.empty_cache()

    # Infer resized pixel count from processor internals if available
    ip = processor.image_processor
    rec = {
        "max_pixels_arg": args.max_pixels,
        "processor_size": getattr(ip, "size", None),
        "processor_max_pixels": getattr(ip, "max_pixels", None),
        "image": args.image,
        "raw_wh": raw_wh,
        "raw_pixels": raw_px,
        "seq_len": seq_len,
        "pixel_values_shape": pv_shape,
        "image_grid_thw": thw_list,
        "n_vision_tokens_est": n_vision,
        "after_load_alloc_gib": after_load_alloc / 1024**3,
        "after_load_reserved_gib": after_load_rsrv / 1024**3,
        "before_fwd_alloc_gib": before / 1024**3,
        "peak_alloc_gib": peak_alloc / 1024**3,
        "peak_reserved_gib": peak_rsrv / 1024**3,
        "after_fwd_alloc_gib": after / 1024**3,
        "delta_peak_minus_load_gib": (peak_alloc - after_load_alloc) / 1024**3,
        "param_bytes_est": sum(p.numel() * p.element_size() for p in base.parameters()),
        "param_gib": sum(p.numel() * p.element_size() for p in base.parameters()) / 1024**3,
        "model_loaded_once": True,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(args.out, "w"), indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
