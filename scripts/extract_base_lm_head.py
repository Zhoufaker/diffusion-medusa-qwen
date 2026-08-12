"""
extract_base_lm_head.py — One-shot script: pull lm_head.weight from
Qwen/Qwen2.5-VL-7B-Instruct on HuggingFace and save it as a single
safetensors file at /scratch/li96/mz9869/medusa_assets/base_lm_head.safetensors.

Why: We only need the lm_head layer to seed the Linked Medusa heads. The full
7B model is ~15 GB which won't fit on gdata; downloading just the one shard
that contains lm_head.weight keeps scratch usage minimal.

Modes:
    --mode index_only   Download only model.safetensors.index.json, locate
                        which shard holds lm_head.weight, print info, exit.
                        (dry run — must run this first to confirm the shard.)
    --mode full         Same as above PLUS download that single shard, extract
                        lm_head.weight via safe_open, convert to fp16 if
                        needed, write the standalone safetensors, then delete
                        the temp shard.

The HF cache is forced into HF_HOME=/scratch/li96/mz9869/tmp_hf_download/
so that downloads never touch the home filesystem (which has tight quota).

Reference (linked_medusa_spec.md §2):
    expected lm_head.weight shape: (151936, 3584)  i.e. (vocab, hidden)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
LM_HEAD_KEY = "lm_head.weight"

# Qwen2.5-VL-7B-Instruct: lm_head physical output dim is 152064 (padded for
# hardware alignment); effective tokenizer vocab is 151936. We assert hidden
# but allow the vocab dim to vary, then record both numbers in metadata.
EXPECTED_HIDDEN = 3584
EFFECTIVE_VOCAB = 151936

DEFAULT_HF_HOME = "/scratch/li96/mz9869/tmp_hf_download"
DEFAULT_OUTPUT = "/scratch/li96/mz9869/medusa_assets/base_lm_head.safetensors"
DEFAULT_METADATA = "/scratch/li96/mz9869/medusa_assets/base_lm_head_metadata.json"


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def setup_hf_env(hf_home: str) -> None:
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = hf_home
    print(f"[env] HF_HOME              = {hf_home}")


def download_index(repo_id: str) -> tuple[str, dict] | tuple[None, None]:
    """Download ONLY the index json. Returns (local_path, parsed_dict).

    If the repo has no index file (i.e. weights live in a single
    model.safetensors with no sharding), returns (None, None) so the caller
    can fall back to single-file mode.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        idx_path = hf_hub_download(repo_id=repo_id, filename=INDEX_FILE)
    except EntryNotFoundError:
        print(f"[index] {INDEX_FILE} not present in repo — model likely uses a single-file shard.")
        return None, None

    with open(idx_path) as f:
        idx = json.load(f)
    return idx_path, idx


def find_lm_head_shard(idx: dict) -> str:
    weight_map = idx.get("weight_map") or {}
    if LM_HEAD_KEY not in weight_map:
        sample_keys = list(weight_map.keys())[:5]
        raise RuntimeError(
            f"{LM_HEAD_KEY!r} not found in index weight_map. "
            f"First 5 keys for context: {sample_keys}. "
            "If this model ties lm_head to embed_tokens, we'd need a different extraction path."
        )
    return weight_map[LM_HEAD_KEY]


def get_remote_file_size(repo_id: str, filename: str) -> int | None:
    """Best-effort remote file size lookup (HEAD request via repo_info)."""
    from huggingface_hub import HfApi
    try:
        info = HfApi().repo_info(repo_id=repo_id, files_metadata=True)
        for sib in info.siblings:
            if sib.rfilename == filename:
                return getattr(sib, "size", None) or getattr(sib, "lfs", {}).get("size")
    except Exception as e:
        print(f"[warn] couldn't fetch remote file size: {e}")
    return None


def cmd_index_only(args) -> int:
    print(f"[step] dry-run: only fetching {INDEX_FILE}")
    idx_path, idx = download_index(args.repo_id)
    if idx is None:
        size = get_remote_file_size(args.repo_id, SINGLE_FILE)
        print(f"[result] no index file — fall back to single-file shard '{SINGLE_FILE}'")
        if size is not None:
            print(f"[result] remote size of {SINGLE_FILE}: {_human_bytes(size)}")
        else:
            print("[result] could not determine remote size of single-file shard")
        print("\n>>> shard to download in --mode full: " + SINGLE_FILE)
        print(">>> WARNING: this is the entire model file (no sharding) — could be ~15 GB.")
        return 0

    shard = find_lm_head_shard(idx)
    print(f"[index] downloaded to: {idx_path}")
    print(f"[index] total tensors in weight_map: {len(idx.get('weight_map') or {})}")
    print(f"[index] {LM_HEAD_KEY} lives in shard: {shard}")

    size = get_remote_file_size(args.repo_id, shard)
    if size is not None:
        print(f"[index] remote size of that shard: {_human_bytes(size)}")
    else:
        print("[index] (couldn't fetch shard size from HF API)")

    total = idx.get("metadata", {}).get("total_size")
    if total:
        print(f"[index] total model size (per index metadata): {_human_bytes(total)}")

    print(f"\n>>> shard to download in --mode full: {shard}")
    return 0


def cmd_full(args) -> int:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    print(f"[step] full extraction starting; output -> {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    idx_path, idx = download_index(args.repo_id)
    if idx is None:
        shard = SINGLE_FILE
        print(f"[full] no index — using single-file '{shard}'")
    else:
        shard = find_lm_head_shard(idx)
        print(f"[full] {LM_HEAD_KEY} -> shard '{shard}'")

    print(f"[full] downloading shard '{shard}' (this may take several minutes)")
    shard_path = hf_hub_download(repo_id=args.repo_id, filename=shard)
    shard_size = Path(shard_path).stat().st_size
    print(f"[full] shard downloaded: {shard_path}  ({_human_bytes(shard_size)})")

    print("[full] opening shard with safe_open and reading lm_head.weight only")
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        if LM_HEAD_KEY not in f.keys():
            keys_sample = list(f.keys())[:8]
            raise RuntimeError(
                f"{LM_HEAD_KEY!r} not found in {shard}. "
                f"Tensors in this shard (first 8): {keys_sample}"
            )
        weight = f.get_tensor(LM_HEAD_KEY)

    print(f"[full] raw shape={tuple(weight.shape)} dtype={weight.dtype}")
    source_dtype_str = str(weight.dtype).replace("torch.", "")

    if weight.ndim != 2:
        raise RuntimeError(f"lm_head.weight must be 2D, got ndim={weight.ndim}")
    vocab_size, hidden_dim = int(weight.shape[0]), int(weight.shape[1])

    if hidden_dim != EXPECTED_HIDDEN:
        raise RuntimeError(
            f"hidden_dim mismatch: got {hidden_dim}, expected {EXPECTED_HIDDEN} "
            f"(Qwen2.5-VL-7B-Instruct hidden size). Aborting — this is a real config drift."
        )

    if vocab_size != EFFECTIVE_VOCAB:
        print(
            f"[info] vocab dim = {vocab_size} (effective tokenizer vocab is {EFFECTIVE_VOCAB}; "
            f"the {vocab_size - EFFECTIVE_VOCAB} extra rows are padding for hardware alignment "
            f"and are kept as-is, no truncation)"
        )

    if weight.dtype != torch.float16:
        print(f"[full] converting dtype {weight.dtype} -> torch.float16 at write time (V100 has no bf16)")
        weight = weight.to(torch.float16)

    weight = weight.contiguous()

    save_file({LM_HEAD_KEY: weight}, args.output)
    out_size = Path(args.output).stat().st_size
    print(f"[full] saved -> {args.output}  ({_human_bytes(out_size)})")

    metadata = {
        "vocab_size": vocab_size,
        "effective_vocab_size": EFFECTIVE_VOCAB,
        "hidden_dim": hidden_dim,
        "dtype": "fp16",
        "source_dtype": source_dtype_str,
        "source_repo": args.repo_id,
        "source_shard": shard,
    }
    Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metadata, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(f"[full] metadata -> {args.metadata}")

    if not args.keep_temp:
        # Delete the downloaded shard from HF cache to free scratch.
        # Don't blow away whole HF_HOME (it might hold the index json which
        # is tiny but harmless to keep). We just delete the shard's blob and
        # any matching snapshot symlink.
        cache_dir = Path(os.environ["HF_HOME"])
        print(f"[cleanup] removing HF cache at {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)
        print("[cleanup] done")

    print("\n=== SUMMARY ===")
    print(f"output path   : {args.output}")
    print(f"metadata path : {args.metadata}")
    print(f"shape         : {tuple(weight.shape)}")
    print(f"dtype         : {weight.dtype}")
    print(f"file size     : {_human_bytes(out_size)}")
    print(f"vocab_size    : {vocab_size}  (effective tokenizer vocab: {EFFECTIVE_VOCAB})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Extract base lm_head.weight from Qwen2.5-VL-7B-Instruct")
    p.add_argument("--mode", choices=("index_only", "full"), required=True,
                   help="index_only = dry run (just locate shard); full = download + extract + cleanup")
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--hf-home", default=DEFAULT_HF_HOME,
                   help="Where HF caches files. Defaults to scratch.")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="Path of the standalone lm_head safetensors file (full mode only).")
    p.add_argument("--metadata", default=DEFAULT_METADATA,
                   help="Path of the JSON sidecar with vocab_size / dtype provenance (full mode only).")
    p.add_argument("--keep-temp", action="store_true",
                   help="Don't delete the HF cache after extraction (full mode).")
    args = p.parse_args()

    setup_hf_env(args.hf_home)

    if args.mode == "index_only":
        return cmd_index_only(args)
    return cmd_full(args)


if __name__ == "__main__":
    sys.exit(main())
