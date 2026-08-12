"""gen_cache_rollout.py — Phase A general-domain cache generator.

Greedy-rollout self-distillation cache for retraining the Linked Medusa heads on
the EVAL domain (LLaVA-Instruct), to isolate v1's chart-domain OOD (cache_source
_audit.py measured a +57pp chart-vs-llava composition gap).

Method (mirrors decode.common.vanilla_greedy, the known-correct image-greedy
path, but records hidden states):
    for each prompt: make_image_inputs -> prefill -> greedy generate <= max_new.
    At every emit step we record the last-layer pre-lm_head hidden that PRODUCED
    the token (hidden_states[-1][0,-1]) and the emitted token id.

Schema (identical to v1 cache, TARGET convention — see train/loss.py):
    {'hidden': (L, 3584) fp16, 'tokens': (L,) int64}
    argmax(base_lm_head @ hidden[t]) == tokens[t]   (head_0 reproduces it ->
    head_0 init loss << 1.0; misalignment would blow this gate up).

Sharding: run K parallel jobs over the SAME --out-dir with
--shard-id i --num-shards K; job i handles prompt indices where idx % K == i.
Each writes a distinct <idx>.pt and skips already-present files (resumable).
Run --manifest-only afterwards to write manifest.json over all shards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decode.common import load_base, make_image_inputs, argmax_masked


@torch.no_grad()
def rollout_one(base, processor, question, image_path, max_new, eos_id, device):
    """Greedy rollout recording (hidden, token) per emit step. TARGET convention."""
    inputs = make_image_inputs(processor, question, image_path, device)
    out = base(**inputs, use_cache=True, output_hidden_states=True)
    past = out.past_key_values
    h = out.hidden_states[-1][0, -1, :]
    tok = argmax_masked(out.logits[0, -1, :])
    hs = [h.detach().to("cpu", torch.float16)]
    toks = [tok]
    del out
    steps = 1
    while steps < max_new and tok != eos_id:
        out = base(input_ids=torch.tensor([[tok]], device=device),
                   past_key_values=past, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        h = out.hidden_states[-1][0, -1, :]
        tok = argmax_masked(out.logits[0, -1, :])
        hs.append(h.detach().to("cpu", torch.float16))
        toks.append(tok)
        del out
        steps += 1
    hidden = torch.stack(hs)                       # (L, 3584) fp16
    tokens = torch.tensor(toks, dtype=torch.int64)  # (L,)
    return hidden, tokens


def write_manifest(out_dir: Path, total_prompts: int) -> None:
    n = sum(1 for p in out_dir.iterdir() if p.suffix == ".pt" and p.stem.isdigit())
    manifest = {"n_cached": n, "n_skipped": 0, "total": total_prompts,
                "complete": n >= total_prompts}
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=2)
    print(f"[manifest] n_cached={n} total={total_prompts} complete={manifest['complete']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="/scratch/li96/mz9869/onpolicy_data/rollout_prompts.json")
    ap.add_argument("--images-dir", default="/scratch/li96/mz9869/onpolicy_data/images")
    ap.add_argument("--out-dir", default="/scratch/li96/mz9869/cached_data/llava_general_35k")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="cap total prompts (smoke)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.load(open(args.prompts))
    if args.limit is not None:
        prompts = prompts[: args.limit]
    total = len(prompts)

    if args.manifest_only:
        write_manifest(out_dir, total)
        return 0

    base, processor = load_base(args.model_id, args.device)
    eos_id = processor.tokenizer.eos_token_id
    images_dir = Path(args.images_dir)

    my_idx = [i for i in range(total) if i % args.num_shards == args.shard_id]
    print(f"[gen] shard {args.shard_id}/{args.num_shards}: {len(my_idx)} prompts "
          f"(of {total}); out={out_dir}", flush=True)

    done = skipped = failed = 0
    t0 = time.time()
    for n, i in enumerate(my_idx):
        fp = out_dir / f"{i}.pt"
        if fp.exists():
            skipped += 1
            continue
        p = prompts[i]
        img = images_dir / p["image"]
        if not img.exists():
            failed += 1
            print(f"[gen] WARN missing image {img} (idx {i})", flush=True)
            continue
        try:
            hidden, tokens = rollout_one(base, processor, p["question"], img,
                                         args.max_new, eos_id, args.device)
        except Exception as e:
            failed += 1
            print(f"[gen] WARN idx {i} failed: {e}", flush=True)
            continue
        torch.save({"hidden": hidden, "tokens": tokens}, fp)
        done += 1
        if done % 50 == 0:
            dt = time.time() - t0
            rate = done / max(1e-6, dt)
            eta = (len(my_idx) - n - 1) / max(1e-6, rate)
            print(f"[gen] {n+1}/{len(my_idx)} done={done} skip={skipped} fail={failed} "
                  f"L_last={tokens.shape[0]} {rate:.2f} prompt/s ETA {eta/60:.0f}min", flush=True)

    print(f"[gen] shard done: generated={done} skipped={skipped} failed={failed} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
    if args.num_shards == 1:
        write_manifest(out_dir, total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
