#!/usr/bin/env python3
"""3a: dump h at target pos via forward hook during ONE vanilla greedy pass.

No separate replay (replay+greedy double-resident caused OOM on deep pos).
Process-isolated: one prompt per invocation.
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
    argmax_masked,
    filter_prompts,
    load_base,
    make_image_inputs,
    mask_phantom_,
)


@torch.no_grad()
def greedy_dump_h_at(base, processor, question, image_path, max_new, eos_id,
                     target_pos: int, device="cuda:0"):
    """Run greedy; capture last-layer hidden (fp16 CPU) that predicts emitted[target_pos].

    pos=0 → prefill last hidden; pos=k>0 → hidden after consuming emitted[k-1].
    Stops once h is captured (no need to finish full greedy unless pos near end).
    """
    torch.cuda.reset_peak_memory_stats()
    inputs = make_image_inputs(processor, question, image_path, device)
    thw = inputs.get("image_grid_thw")
    thw_list = thw.tolist() if thw is not None else None
    seq_len = int(inputs["input_ids"].shape[-1])
    n_vision = int(thw.prod().item() // 4) if thw is not None else None

    # Always request hidden on prefill; keep only if target_pos==0
    out = base(**inputs, use_cache=True, output_hidden_states=True)
    past = out.past_key_values
    h_last = out.hidden_states[-1][0, -1, :].detach().half().cpu()
    logits = mask_phantom_(out.logits[0, -1, :])
    del out
    torch.cuda.empty_cache()

    if target_pos == 0:
        peak = torch.cuda.max_memory_allocated() / 1024**3
        peak_r = torch.cuda.max_memory_reserved() / 1024**3
        del past, inputs
        return h_last, [], {
            "seq_len": seq_len, "image_grid_thw": thw_list,
            "n_vision_tokens_est": n_vision,
            "peak_alloc_gib": peak, "peak_reserved_gib": peak_r,
        }

    nxt = argmax_masked(logits)
    emitted = [nxt]
    # Need hidden after consuming emitted[target_pos - 1]
    while len(emitted) < max_new:
        need_h = (len(emitted) == target_pos)
        out = base(
            input_ids=torch.tensor([[nxt]], device=device),
            past_key_values=past, use_cache=True,
            output_hidden_states=need_h,
        )
        past = out.past_key_values
        if need_h:
            h_last = out.hidden_states[-1][0, -1, :].detach().half().cpu()
            del out
            break
        logits = mask_phantom_(out.logits[0, -1, :])
        del out
        nxt = argmax_masked(logits)
        if nxt == eos_id:
            emitted.append(nxt)
            break
        emitted.append(nxt)
        if len(emitted) % 32 == 0:
            torch.cuda.empty_cache()

    peak = torch.cuda.max_memory_allocated() / 1024**3
    peak_r = torch.cuda.max_memory_reserved() / 1024**3
    del past, inputs
    torch.cuda.empty_cache()
    return h_last, emitted, {
        "seq_len": seq_len, "image_grid_thw": thw_list,
        "n_vision_tokens_est": n_vision,
        "peak_alloc_gib": peak, "peak_reserved_gib": peak_r,
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

    print(f"[3a-hook] id={args.prompt_id} pos={args.pos}")
    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    eos = processor.tokenizer.eos_token_id
    from PIL import Image
    im = Image.open(images / p["image"]).convert("RGB")
    h, emitted, mem = greedy_dump_h_at(
        base, processor, p["question"], images / p["image"],
        args.max_new_tokens, eos, args.pos,
    )
    assert h is not None and h.dtype == torch.float16
    h_path = h_dir / f"{args.prompt_id}_pos{args.pos}.pt"
    torch.save({"h": h, "prompt_id": str(args.prompt_id), "pos": args.pos,
                "method": "greedy_hook_single_pass",
                "emitted_prefix_len": len(emitted)}, h_path)
    meta = {
        "prompt_id": str(args.prompt_id), "pos": args.pos,
        "h_path": str(h_path), "h_shape": list(h.shape),
        "raw_wh": list(im.size), "raw_pixels": im.size[0] * im.size[1],
        "method": "greedy_hook_single_pass", **mem,
    }
    meta_path = out_dir / f"item3_h_meta_{args.prompt_id}_pos{args.pos}.json"
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"  wrote {h_path} peak_alloc={mem['peak_alloc_gib']:.3f}GiB "
          f"seq={mem['seq_len']} thw={mem['image_grid_thw']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
