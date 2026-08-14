"""extract_ctx_features.py — W1: 5-layer context feature extraction (DFlash line).

For each locked (prompt, greedy rollout) pair in the 34,999 self-distillation
cache, run ONE teacher-forcing forward of Qwen2.5-VL-7B over the full sequence
(vision + prompt text + rollout tokens) and record the hidden states of target
layers build_target_layer_ids(28, 5) -> [1, 7, 13, 19, 25], stacked as
(5, T_total, 3584) bf16.  Plan-A: vision tokens ARE part of the context.

Provenance contract (W1 spec):
  * Preprocessing mirrors scripts/gen_cache_rollout.py PARAMETER-FOR-PARAMETER:
    same decode.common.load_base (fp16, attn=sdpa), same make_image_inputs
    (chat template, add_generation_prompt=True), same processor DEFAULT pixel
    settings (gen_cache_rollout.py never called apply_max_pixels — the eval-side
    501760 cap does NOT apply here).  Runtime-read values are frozen into
    PREPROCESS_SPEC and written to the manifest.
  * Old cache .pt files are opened READ-ONLY; their `tokens` field drives
    teacher forcing and is fingerprinted (sha256) into the manifest.
  * Inline exactness audit (TARGET convention drift detection): at every
    rollout position, argmax_masked(logits) is compared with the old cached
    token.  Pre-registered gates (full run): sample-exact >= 99.5%,
    position-level >= 99.9%.  Near-tie mismatches (fp16 tie flips) are
    reported separately via the logit gap to the expected token.

Sharding: safetensors, SHARD_SIZE=256 samples/shard, shard order == old cache
index order (deterministic).  Keys per sample i: "{i}.ctx" (5,T,3584) bf16,
"{i}.ids" (T,) int64, "{i}.spans" (6,) int64
[vision_s, vision_e, prompt_s, prompt_e, rollout_s, rollout_e)  (prompt span =
full prefill [0,P) including vision; rollout span = [P, P+L)).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from safetensors.torch import save_file

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from decode.common import (  # noqa: E402
    EFFECTIVE_VOCAB,
    load_base,
    make_image_inputs,
    mask_phantom_,
)

sys.path.insert(0, str(_ROOT / "external" / "dflash"))
from dflash.model import build_target_layer_ids  # noqa: E402

SHARD_SIZE = 256
NUM_TARGET_LAYERS = 28
NUM_DRAFT_LAYERS = 5
TARGET_LAYER_IDS = build_target_layer_ids(NUM_TARGET_LAYERS, NUM_DRAFT_LAYERS)
# hidden_states tuple index 0 is the embedding output -> layer l lives at l+1.
HIDDEN_TUPLE_IDS = [l + 1 for l in TARGET_LAYER_IDS]

# Mirrors gen_cache_rollout.py / decode.common.load_base exactly.  Fields whose
# value is only known at runtime (processor pixel settings, token ids) are
# filled in by freeze_preprocess_spec() and asserted non-None before any write.
PREPROCESS_SPEC = {
    "mirror_of": "scripts/gen_cache_rollout.py",
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "forward_dtype": "float16",          # load_base: torch.float16 (rollout parity)
    "attn_implementation": "sdpa",       # load_base asserts sdpa
    "storage_dtype": "bfloat16",         # W1 spec 1c
    "chat_template": "processor.apply_chat_template(add_generation_prompt=True)",
    "message_schema": "[{role:user, content:[{type:image},{type:text,text:question}]}]",
    "processor_padding": True,           # processor(text=[...], images=[...], padding=True)
    "max_pixels_override": None,         # gen_cache_rollout applied NO apply_max_pixels
    "runtime_max_pixels": None,          # frozen at runtime from processor
    "runtime_min_pixels": None,
    "effective_vocab": EFFECTIVE_VOCAB,  # argmax over [0, 151936)
    "rollout_max_new": 256,              # gen_cache_rollout --max-new default
    "eos_token_id": None,                # frozen at runtime
    "image_pad_token_id": None,          # frozen at runtime
    "target_layer_ids": TARGET_LAYER_IDS,
    "hidden_tuple_indices": HIDDEN_TUPLE_IDS,
    "old_cache_gen_hardware": "gpuvolta V100 fp16 sdpa (jobs 171434109[0-7], 2026-06-17)",
    "extract_hardware": "dgxa100 A100 fp16 sdpa",
}

NEAR_TIE_GAP = 0.05  # |logit(top1) - logit(expected)| below this => near-tie bucket


def freeze_preprocess_spec(processor) -> dict:
    spec = dict(PREPROCESS_SPEC)
    ip = processor.image_processor
    # transformers 5.x fast processor keeps pixel caps in ip.size
    # ({shortest_edge, longest_edge}); legacy attrs may be None.
    spec["runtime_max_pixels"] = getattr(ip, "max_pixels", None)
    spec["runtime_min_pixels"] = getattr(ip, "min_pixels", None)
    spec["runtime_size"] = dict(getattr(ip, "size", None) or {})
    spec["runtime_image_processor_class"] = type(ip).__name__
    spec["eos_token_id"] = int(processor.tokenizer.eos_token_id)
    spec["image_pad_token_id"] = int(
        processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    )
    assert spec["max_pixels_override"] is None, "extraction must NOT cap pixels"
    for k in ("eos_token_id", "image_pad_token_id"):
        assert spec[k] is not None, f"PREPROCESS_SPEC field {k} unresolved"
    return spec


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def find_vision_span(ids: torch.Tensor, image_pad_id: int) -> tuple[int, int]:
    """[start, end) of the contiguous <|image_pad|> run; (0, 0) if no image."""
    pos = (ids == image_pad_id).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        return 0, 0
    s, e = int(pos[0]), int(pos[-1]) + 1
    assert pos.numel() == e - s, f"non-contiguous vision span ({pos.numel()} vs {e - s})"
    return s, e


@torch.no_grad()
def extract_one(base, spec: dict, inputs, old_tokens: torch.Tensor, device: str,
                audit_only: bool = False):
    """Teacher-forcing forward over prompt + rollout; returns tensors + audit."""
    prompt_ids = inputs["input_ids"]                       # (1, P)
    P = prompt_ids.shape[1]
    L = old_tokens.shape[0]
    full_ids = torch.cat(
        [prompt_ids, old_tokens.view(1, -1).to(device)], dim=1
    )                                                      # (1, T), T = P + L
    out = base(
        input_ids=full_ids,
        attention_mask=torch.ones_like(full_ids),
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        output_hidden_states=True,
        use_cache=False,
    )
    ctx = torch.stack(
        [out.hidden_states[t][0] for t in HIDDEN_TUPLE_IDS]
    )                                                      # (5, T, 3584) fp16 gpu
    # --- inline exactness audit (TARGET convention) ---
    # logits at position P-1+j predict old_tokens[j], j in [0, L)
    logits = mask_phantom_(out.logits[0, P - 1 : P + L - 1, :].float())  # (L, V)
    pred = logits.argmax(-1)                               # (L,)
    exp = old_tokens.to(device)
    match = pred == exp
    n_match = int(match.sum())
    mismatch = []
    if n_match < L:
        for j in (~match).nonzero(as_tuple=True)[0][:16].tolist():
            gap = float(logits[j, pred[j]] - logits[j, exp[j]])
            mismatch.append({"pos": j, "gap": round(gap, 6)})
    ctx_cpu = None if audit_only else ctx.to("cpu", torch.bfloat16).contiguous()
    ids_cpu = full_ids[0].to("cpu", torch.int64).contiguous()
    pred_cpu = pred.to("cpu", torch.int64) if audit_only else None
    del out, ctx, logits
    return ctx_cpu, ids_cpu, P, L, n_match, mismatch, pred_cpu


def flush_shard(out_dir: Path, shard_id: int, buf: dict, partial: bool) -> str:
    name = f"shard_{shard_id:05d}.safetensors"
    tensors = {}
    for i, (ctx, ids, spans) in sorted(buf.items()):
        tensors[f"{i}.ctx"] = ctx
        tensors[f"{i}.ids"] = ids
        tensors[f"{i}.spans"] = spans
    save_file(tensors, str(out_dir / name), metadata={"partial": str(partial)})
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="/scratch/li96/mz9869/onpolicy_data/rollout_prompts.json")
    ap.add_argument("--images-dir", default="/scratch/li96/mz9869/onpolicy_data/images")
    ap.add_argument("--old-cache-dir", default="/scratch/li96/mz9869/cached_data/llava_general_35k")
    ap.add_argument("--out-dir", default="/scratch/li96/mz9869/dflash_data/ctx_cache_35k")
    ap.add_argument("--model-id", default=PREPROCESS_SPEC["model_id"])
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--end-idx", type=int, default=None, help="exclusive; default = all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--manifest-name", default="ctx_manifest.json")
    ap.add_argument("--index-file", default=None,
                    help="JSON list of indices; overrides --start/--end-idx")
    ap.add_argument("--audit-only", action="store_true",
                    help="no feature shards; save per-sample TF pred sequences")
    args = ap.parse_args()

    if not (args.index_file or args.audit_only):
        assert args.start_idx % SHARD_SIZE == 0, "start-idx must be shard-aligned"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.load(open(args.prompts))
    total = len(prompts)
    end = total if args.end_idx is None else min(args.end_idx, total)
    images_dir = Path(args.images_dir)
    old_dir = Path(args.old_cache_dir)

    base, processor = load_base(args.model_id, args.device)
    spec = freeze_preprocess_spec(processor)
    image_pad_id = spec["image_pad_token_id"]

    indices = (json.load(open(args.index_file)) if args.index_file
               else list(range(args.start_idx, end)))
    records: dict[str, dict] = {}
    shard_names: dict[int, str] = {}
    buf: dict[int, tuple] = {}
    preds_out: dict[int, torch.Tensor] = {}
    n_pos_total = n_match_total = n_sample_exact = n_done = n_failed = 0
    t0 = time.time()
    tok_count = 0

    for i in indices:
        shard_id = i // SHARD_SIZE
        p = prompts[i]
        img = images_dir / p["image"]
        try:
            old = torch.load(old_dir / f"{i}.pt", map_location="cpu", weights_only=True)
            old_tokens = old["tokens"]
            sha = hashlib.sha256(old_tokens.numpy().tobytes()).hexdigest()
            inputs = make_image_inputs(processor, p["question"], img, args.device)
            ctx, ids, P, L, n_match, mismatch, pred = extract_one(
                base, spec, inputs, old_tokens, args.device,
                audit_only=args.audit_only,
            )
        except Exception as e:  # noqa: BLE001 — per-sample fault isolation, counted + gated below
            n_failed += 1
            print(f"[ext] WARN idx {i} failed: {e}", flush=True)
            records[str(i)] = {"idx": i, "failed": True, "error": str(e)[:200]}
            continue
        vs, ve = find_vision_span(ids[:P], image_pad_id)
        spans = torch.tensor([vs, ve, 0, P, P, P + L], dtype=torch.int64)
        if args.audit_only:
            preds_out[i] = pred
        else:
            buf[i] = (ctx, ids, spans)
        records[str(i)] = {
            "idx": i, "shard": shard_id, "T": P + L, "P": P, "L": L,
            "spans": {"vision": [vs, ve], "prompt": [0, P], "rollout": [P, P + L]},
            "old_tokens_sha256": sha, "n_pos": L, "n_match": n_match,
            "sample_exact": n_match == L, "mismatch_head16": mismatch,
        }
        n_pos_total += L
        n_match_total += n_match
        n_sample_exact += int(n_match == L)
        n_done += 1
        tok_count += P + L
        if not args.audit_only and ((i + 1) % SHARD_SIZE == 0 or (i + 1) == end):
            partial = (i + 1) % SHARD_SIZE != 0 or (i % SHARD_SIZE + 1) != len(buf)
            shard_names[shard_id] = flush_shard(out_dir, shard_id, buf, partial)
            buf = {}
        if n_done and n_done % 50 == 0:
            dt = time.time() - t0
            print(f"[ext] {i + 1 - args.start_idx}/{end - args.start_idx} "
                  f"done={n_done} fail={n_failed} "
                  f"{n_done / dt * 60:.1f} samples/min {tok_count / dt:.0f} tok/s "
                  f"posmatch={n_match_total / max(1, n_pos_total):.6f}", flush=True)

    dt = time.time() - t0
    peak_gb = (torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0
    agg = {
        "n_done": n_done, "n_failed": n_failed,
        "sample_exact_rate": n_sample_exact / max(1, n_done),
        "pos_match_rate": n_match_total / max(1, n_pos_total),
        "gate_sample_exact_ge": 0.995, "gate_pos_match_ge": 0.999,
        "gate_sample_pass": (n_sample_exact / max(1, n_done)) >= 0.995,
        "gate_pos_pass": (n_match_total / max(1, n_pos_total)) >= 0.999,
        "throughput_samples_per_min": n_done / dt * 60,
        "throughput_tok_per_s": tok_count / dt,
        "peak_gpu_mem_gib": round(peak_gb, 2),
        "wall_s": round(dt, 1),
    }
    if args.audit_only and preds_out:
        torch.save(preds_out, out_dir / "tf_preds.pt")
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code_commit": git_commit_hash(),
        "hardware": os.environ.get(
            "EXTRACT_HARDWARE_TAG",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        ),
        "audit_only": bool(args.audit_only),
        "index_file": args.index_file,
        "range": [args.start_idx, end], "total_prompts": total,
        "shard_size": SHARD_SIZE, "shards": shard_names,
        "preprocess_spec": spec, "near_tie_gap": NEAR_TIE_GAP,
        "aggregate": agg, "samples": records,
    }
    mpath = out_dir / args.manifest_name
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"[ext] DONE range=[{args.start_idx},{end}) done={n_done} fail={n_failed}\n"
          f"[ext] sample_exact={agg['sample_exact_rate']:.6f} "
          f"pos_match={agg['pos_match_rate']:.6f} "
          f"(gates: >=0.995 / >=0.999 -> {agg['gate_sample_pass']}/{agg['gate_pos_pass']})\n"
          f"[ext] {agg['throughput_samples_per_min']:.1f} samples/min "
          f"{agg['throughput_tok_per_s']:.0f} tok/s peak_mem={peak_gb:.1f}GiB\n"
          f"[ext] manifest -> {mpath}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
