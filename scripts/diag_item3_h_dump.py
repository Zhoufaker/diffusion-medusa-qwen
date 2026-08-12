#!/usr/bin/env python3
"""3a offline: dump h (fp16) at audit positions → CPU lm_head top-3; item5 mem sidecar.

GPU only does causal replay with output_hidden_states on the last step, then
writes h to disk and frees. lm_head matmul is CPU fp32. Does not touch decode/.
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
    mask_phantom_,
    vanilla_greedy,
)
from scripts.diag_prefill_mem import apply_max_pixels  # noqa: E402


def replay_hidden(base, processor, question, image_path, prefix):
    device = "cuda:0"
    inputs = make_image_inputs(processor, question, image_path, device)
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
    del past, inputs
    torch.cuda.empty_cache()
    return h


def mem_probe(base, processor, image, question, max_pixels, out_path: Path):
    apply_max_pixels(processor, max_pixels)
    torch.cuda.reset_peak_memory_stats()
    after_load_alloc = torch.cuda.memory_allocated()
    after_load_rsrv = torch.cuda.memory_reserved()
    from PIL import Image
    im = Image.open(image).convert("RGB")
    inputs = make_image_inputs(processor, question, Path(image), "cuda:0")
    seq_len = int(inputs["input_ids"].shape[-1])
    thw = inputs.get("image_grid_thw")
    thw_list = thw.tolist() if thw is not None else None
    n_vision = int(thw.prod().item() // 4) if thw is not None else None
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = base(**inputs, use_cache=True)
    rec = {
        "max_pixels_arg": max_pixels,
        "image": str(image),
        "raw_wh": list(im.size),
        "raw_pixels": im.size[0] * im.size[1],
        "seq_len": seq_len,
        "image_grid_thw": thw_list,
        "n_vision_tokens_est": n_vision,
        "after_load_alloc_gib": after_load_alloc / 1024**3,
        "after_load_reserved_gib": after_load_rsrv / 1024**3,
        "before_fwd_alloc_gib": before / 1024**3,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "after_fwd_alloc_gib": torch.cuda.memory_allocated() / 1024**3,
    }
    del out, inputs
    torch.cuda.empty_cache()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(out_path, "w"), indent=2)
    print(f"[item5] wrote {out_path} peak_alloc={rec['peak_alloc_gib']:.3f}GiB "
          f"seq={seq_len} thw={thw_list}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="/scratch/li96/mz9869/medusa_outputs/"
                    "linked_medusa_c1_eagle/diag_triad/item3_fp32_audit.json")
    ap.add_argument("--out-dir", default="/scratch/li96/mz9869/medusa_outputs/"
                    "linked_medusa_c1_eagle/diag_triad")
    ap.add_argument("--manifest", default="/scratch/li96/mz9869/eval_manifests/manifest_300.json")
    ap.add_argument("--images-dir", default="/g/data/li96/mz9869/data/coco_subset")
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--skip-mem", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    h_dir = out_dir / "item3_h_fp16"
    h_dir.mkdir(parents=True, exist_ok=True)
    audit = json.load(open(args.audit))
    by_id = {p["id"]: p for p in filter_prompts(
        args.manifest, 80, 42, ordered=True)}
    images = Path(args.images_dir)

    print("[3a] loading base")
    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    eos_id = processor.tokenizer.eos_token_id
    meta = []

    for r in audit:
        pid = r["prompt_id"]
        pos = int(r["pos"])
        p = by_id[pid]
        print(f"[3a] dump h id={pid} pos={pos}")
        torch.cuda.empty_cache()
        g = vanilla_greedy(base, processor, p["question"], images / p["image"],
                           args.max_new_tokens, eos_id)
        prefix = g[:pos]
        h = replay_hidden(base, processor, p["question"], images / p["image"], prefix)
        assert h is not None and h.dtype == torch.float16
        h_path = h_dir / f"{pid}_pos{pos}.pt"
        torch.save({"h": h, "prompt_id": pid, "pos": pos, "prefix_len": len(prefix)},
                   h_path)
        meta.append({
            "prompt_id": pid, "pos": pos, "h_path": str(h_path),
            "h_shape": list(h.shape), "tree_emitted_tok": r["tree_emitted_tok"],
            "fp16_causal_eq_fp32": r["fp16_causal_eq_fp32"],
            "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        })
        print(f"  wrote {h_path} shape={tuple(h.shape)} "
              f"alloc={meta[-1]['allocated_gib']:.2f}GiB")
        del h, g
        torch.cuda.empty_cache()
        json.dump(meta, open(out_dir / "item3_h_dump_meta.json", "w"), indent=2)

    # CPU top-3 from dumps (lm_head on CPU; base still on GPU but weight copied)
    print("[3a] CPU lm_head.float() @ h → top-3")
    lm_w = base.lm_head.weight.detach().float().cpu()
    out_rows = []
    outp = out_dir / "item3_fp32_top3.json"
    for r, m in zip(audit, meta):
        blob = torch.load(m["h_path"], map_location="cpu", weights_only=True)
        h = blob["h"].float()
        logits = torch.nn.functional.linear(h, lm_w)
        logits = mask_phantom_(logits)
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
            "method": "h_dump_fp16 + lm_head.weight.float() @ h (CPU)",
        }
        out_rows.append(rec)
        print(f"  {r['prompt_id']} pos={r['pos']} tree={tree} top3={ids} "
              f"in_top2={rec['tree_in_top2']} eq16=32={r['fp16_causal_eq_fp32']}")
        json.dump(out_rows, open(outp, "w"), indent=2)

    eq_rows = [r for r in out_rows if r["fp16_causal_eq_fp32"]]
    fail_top2 = [r for r in eq_rows if not r["tree_in_top2"]]
    print(f"[3a] 16=32=Y rows={len(eq_rows)}; tree∉top-2 among them={len(fail_top2)}")
    for r in fail_top2:
        print(f"  FAIL {r['prompt_id']} tree={r['tree_emitted_tok']} top3={r['fp32_top3_ids']}")

    if not args.skip_mem:
        mmvet = Path("/scratch/li96/mz9869/ood_eval/mmvet/mm-vet/images")
        for px in (501760, 1003520):
            for name in ("v1_82.png", "v1_21.png"):
                img = mmvet / name
                if not img.exists():
                    print(f"[item5] skip missing {img}")
                    continue
                try:
                    mem_probe(
                        base, processor, img, "Describe the image.", px,
                        out_dir / f"item5_mem_{name.replace('.png','')}_{px}.json",
                    )
                except Exception as e:
                    print(f"[item5] FAIL {name} px={px}: {e!r}")
                    json.dump(
                        {"error": repr(e), "max_pixels": px, "image": str(img)},
                        open(out_dir / f"item5_mem_{name.replace('.png','')}_{px}_ERR.json", "w"),
                        indent=2,
                    )

    if fail_top2:
        return 2
    if len(out_rows) < len(audit):
        return 1
    print("[3a] DONE all rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
