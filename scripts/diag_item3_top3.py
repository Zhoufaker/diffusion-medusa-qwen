#!/usr/bin/env python3
"""Item3收尾: fp32 top-3 for rows where fp16_causal==fp32; check tree ∈ top-2/top-3."""
from __future__ import annotations

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


def replay_hidden(base, processor, question, image_path, prefix):
    """Prefill + decode prefix; only last step requests hidden states."""
    device = "cuda:0"
    inputs = make_image_inputs(processor, question, image_path, device)
    need_h = len(prefix) == 0
    out = base(**inputs, use_cache=True, output_hidden_states=need_h)
    past = out.past_key_values
    if need_h:
        h = out.hidden_states[-1][0, -1, :].detach().clone()
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
            h = out.hidden_states[-1][0, -1, :].detach().clone()
        del out
    del past, inputs
    torch.cuda.empty_cache()
    return h


def main():
    audit = json.load(open(
        "/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/diag_triad/item3_fp32_audit.json"))
    rows = [r for r in audit if r["fp16_causal_eq_fp32"]]
    print(f"[top3] {len(rows)} rows with fp16_causal==fp32")
    base, processor = load_base("Qwen/Qwen2.5-VL-7B-Instruct")
    lm_w = base.lm_head.weight.detach().float().cpu()
    eos_id = processor.tokenizer.eos_token_id
    images = Path("/g/data/li96/mz9869/data/coco_subset")
    by_id = {p["id"]: p for p in filter_prompts(
        "/scratch/li96/mz9869/eval_manifests/manifest_300.json", 80, 42, ordered=True)}
    out_rows = []
    outp = Path("/scratch/li96/mz9869/medusa_outputs/linked_medusa_c1_eagle/diag_triad/item3_fp32_top3.json")
    for r in rows:
        p = by_id[r["prompt_id"]]
        torch.cuda.empty_cache()
        g = vanilla_greedy(base, processor, p["question"], images / p["image"], 150, eos_id)
        prefix = g[: r["pos"]]
        torch.cuda.empty_cache()
        h = replay_hidden(base, processor, p["question"], images / p["image"], prefix)
        logits = torch.nn.functional.linear(h.float().cpu(), lm_w)
        logits = mask_phantom_(logits)
        top3 = torch.topk(logits, 3)
        ids = [int(x) for x in top3.indices.tolist()]
        vals = [float(x) for x in top3.values.tolist()]
        tree = r["tree_emitted_tok"]
        rec = {
            **{k: r[k] for k in ("prompt_id", "band", "pos", "tree_emitted_tok",
                                 "greedy_emitted_tok", "fp32_argmax", "fp32_top2_gap")},
            "fp32_top3_ids": ids,
            "fp32_top3_logits": vals,
            "tree_in_top2": tree in ids[:2],
            "tree_in_top3": tree in ids[:3],
        }
        out_rows.append(rec)
        print(f"  {r['prompt_id']} pos={r['pos']} tree={tree} top3={ids} "
              f"in_top2={rec['tree_in_top2']} in_top3={rec['tree_in_top3']}")
        json.dump(out_rows, open(outp, "w"), indent=2)  # incremental
        del h, logits, g
        torch.cuda.empty_cache()

    # item5 mem after top3 (same job)
    try:
        from scripts.diag_item5_mem import main as _mem_main  # noqa: F401
        import scripts.diag_item5_mem as mem
        # call via subprocess-like: set argv
        sys.argv = [
            "diag_item5_mem",
            "--max-pixels", "501760",
            "--out", str(outp.parent / "item5_mem_v1_82_501760.json"),
        ]
        mem.main()
    except Exception as e:
        print(f"[item5 mem] skipped/failed: {e!r}")

    outside = [r for r in out_rows if not r["tree_in_top3"]]
    print(f"[done] outside_top3={len(outside)} n={len(out_rows)}")
    if outside:
        print("STOP: tree token outside fp32 top-3:")
        for r in outside:
            print(json.dumps(r))
        return 2
    if len(out_rows) < len(rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
