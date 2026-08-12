"""collate_fn — pad variable-length samples to batch max length.

Spec ref: linked_medusa_spec.md §6.3.

Returns:
    {
        "hidden":         Tensor(B, L_max, H) float16, padded with 0.0
        "tokens":         Tensor(B, L_max)    int64,   padded with -100
        "attention_mask": Tensor(B, L_max)    bool,    True for real tokens
    }

The -100 padding for tokens is the standard CrossEntropyLoss `ignore_index`,
so compute_loss can treat padded positions transparently. attention_mask is
included for downstream masked operations (e.g. evaluation metrics) even
though the basic loss does not need it.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

IGNORE_INDEX = -100


def collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    if not batch:
        raise ValueError("collate_fn received empty batch")

    hiddens = [b["hidden"] for b in batch]
    tokens = [b["tokens"] for b in batch]

    B = len(batch)
    H = hiddens[0].shape[-1]
    L_max = max(h.shape[0] for h in hiddens)

    # Sanity: enforce uniform hidden dim and matching lengths within each sample.
    for h, t in zip(hiddens, tokens):
        if h.shape[-1] != H:
            raise ValueError(f"inconsistent hidden_dim in batch: {h.shape[-1]} vs {H}")
        if h.shape[0] != t.shape[0]:
            raise ValueError(f"hidden vs tokens length mismatch: {h.shape[0]} vs {t.shape[0]}")

    out_hidden = torch.zeros(B, L_max, H, dtype=hiddens[0].dtype)
    out_tokens = torch.full((B, L_max), IGNORE_INDEX, dtype=torch.int64)
    out_mask = torch.zeros(B, L_max, dtype=torch.bool)

    for i, (h, t) in enumerate(zip(hiddens, tokens)):
        L = h.shape[0]
        out_hidden[i, :L] = h
        out_tokens[i, :L] = t
        out_mask[i, :L] = True

    return {
        "hidden": out_hidden,
        "tokens": out_tokens,
        "attention_mask": out_mask,
    }
