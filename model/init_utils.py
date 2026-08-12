"""Helpers for loading the base lm_head weight from disk.

Spec ref: linked_medusa_spec.md §2 / §5.3.

We keep this in its own file so it's clear what the only valid source of
lm_head init is: the safetensors file produced by
scripts/extract_base_lm_head.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch
from safetensors.torch import safe_open


LM_HEAD_KEY = "lm_head.weight"
DEFAULT_LM_HEAD_PATH = "/scratch/li96/mz9869/medusa_assets/base_lm_head.safetensors"
DEFAULT_METADATA_PATH = "/scratch/li96/mz9869/medusa_assets/base_lm_head_metadata.json"


def load_base_lm_head_weight(
    path: str = DEFAULT_LM_HEAD_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
    expected_shape: Tuple[int, int] | None = None,
) -> torch.Tensor:
    """Load the standalone lm_head safetensors written by extract_base_lm_head.py.

    Returns a CPU tensor with the same dtype as on disk (fp16 by spec).

    If `metadata_path` exists, we cross-check vocab_size / hidden_dim with it
    and warn (not error) on mismatch — the metadata is authoritative provenance.
    `expected_shape`, if given, is asserted hard.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"base lm_head not found at {path}. "
            "Run scripts/extract_base_lm_head.py --mode full first."
        )

    with safe_open(str(p), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if LM_HEAD_KEY not in keys:
            raise RuntimeError(
                f"{LM_HEAD_KEY!r} not in {path}; available keys: {keys}"
            )
        weight = f.get_tensor(LM_HEAD_KEY)

    if expected_shape is not None and tuple(weight.shape) != tuple(expected_shape):
        raise ValueError(
            f"lm_head shape mismatch: file={tuple(weight.shape)} expected={tuple(expected_shape)}"
        )

    # Per spec §3 / §9.4, the on-disk base lm_head should be fp16 to match
    # the fp16 training pipeline (V100 has no bf16). Warn but do not raise
    # if the dtype is something else — the caller can still copy_() with
    # PyTorch handling the cast, so it's not fatal.
    if weight.dtype != torch.float16:
        print(
            f"[load_base_lm_head_weight] WARN: expected dtype torch.float16, "
            f"got {weight.dtype}. Training pipeline is fp16; the weight will "
            f"be cast at copy_() time, but you should regenerate the file via "
            f"scripts/extract_base_lm_head.py --mode full to avoid surprises."
        )

    meta_p = Path(metadata_path)
    if meta_p.is_file():
        with open(meta_p) as mf:
            meta = json.load(mf)
        if (meta.get("vocab_size"), meta.get("hidden_dim")) != tuple(weight.shape):
            print(
                f"[load_base_lm_head_weight] WARN: metadata vocab/hidden "
                f"({meta.get('vocab_size')}, {meta.get('hidden_dim')}) != "
                f"file shape {tuple(weight.shape)}"
            )

    return weight
