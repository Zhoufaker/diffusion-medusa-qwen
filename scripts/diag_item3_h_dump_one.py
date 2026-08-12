#!/usr/bin/env python3
"""3a one-shot: single prompt → replay to pos → dump h (fp16) + mem trio → exit.

Process-isolated: designed to be invoked once per prompt so GPU state dies with
the process. Does not touch decode/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from decode.common import (  # noqa: E402
    filter_prompts,
    load_base,
    make_image_inputs,
    vanilla_greedy,
)


def replay_hidden(base, processor, question, image_path, prefix):
    device = "cuda:0"
    torch.cuda.reset_peak_memory_stats()
    inputs = make_image_inputs(processor, question, image_path, device)
    thw = inputs.get("image_grid_thw")
    thw_list = thw.tolist() if thw is not None else None
    seq_len = int(inputs["input_ids"].shape[-1])
    # resize estimate from vision tokens
    n_vision = int(thw.prod().item() // 4) if thw is not None else None
    need_h = len(prefix) == 0
    out = base(**inputs, use_cache=True, output_hidden_states=need_h)
    past = out.past_key_values
    h = None
    if need_h:
        h = out.hidden_states[-1][0, -1, :].detach().half().cpu()
    del out
    torch.cuda.empty_cache()
    for i, tok in enumerate(prefix):
        last = i == len(prefix) - 1
        out = base(
            input_ids=torch.tensor([[tok]], device=device),
            past_key_values=past, use_cache=True,
            output_hidden_states=last,
        )
        past = out.past_key_values
        if last:
            h = out.hidden_states[-1][0, -1, :].detach().half().cpu()
        del out
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_rsrv = torch.cuda.max_memory_reserved() / 1024**3
    del past, inputs
    torch.cuda.empty_cache()
    return h, {
        "seq_len": seq_len,
        "image_grid_thw": thw_list,
        "n_vision_tokens_est": n_vision,
        "peak_alloc_gib": peak_alloc,
        "peak_reserved_gib": peak_rsrv,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--pos", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", default="/scratch/li96/mz9869/eval_manifests/manifest_300.json")
    ap.add_argument("--images-dir", default="/g/data/li96/mz9869/data/coco_subset")
    ap.add_argument("--max-new-tokens", type=int, default=150)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    h_dir = out_dir / "item3_h_fp16"
    h_dir.mkdir(parents=True, exist_ok=True)

    by_id = {str(p["id"]): p for p in filter_prompts(args.manifest, 80, 42, ordered=True)}
    p = by_id[str(args.prompt_id)]
    images = Path(args.images_dir)

    print(f"[3a-one] id={args.prompt_id} pos={args.pos}")
    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    eos = processor.tokenizer.eos_token_id
    g = vanilla_greedy(base, processor, p["question"], images / p["image"],
                       args.max_new_tokens, eos)
    prefix = g[: args.pos]
    from PIL import Image
    im = Image.open(images / p["image"]).convert("RGB")
    raw_wh = list(im.size)
    h, mem = replay_hidden(base, processor, p["question"], images / p["image"], prefix)
    assert h is not None and h.dtype == torch.float16
    h_path = h_dir / f"{args.prompt_id}_pos{args.pos}.pt"
    torch.save({"h": h, "prompt_id": str(args.prompt_id), "pos": args.pos,
                "prefix_len": len(prefix)}, h_path)
    meta = {
        "prompt_id": str(args.prompt_id),
        "pos": args.pos,
        "h_path": str(h_path),
        "h_shape": list(h.shape),
        "raw_wh": raw_wh,
        "raw_pixels": raw_wh[0] * raw_wh[1],
        **mem,
        "allocated_after_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_after_gib": torch.cuda.memory_reserved() / 1024**3,
    }
    meta_path = out_dir / f"item3_h_meta_{args.prompt_id}_pos{args.pos}.json"
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"  wrote {h_path} peak_alloc={mem['peak_alloc_gib']:.3f}GiB "
          f"thw={mem['image_grid_thw']} seq={mem['seq_len']}")
    print(f"  meta {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
