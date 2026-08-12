"""compute_loss — multi-head CE with the correct per-head offset.

Spec ref: linked_medusa_spec.md §6.4.

Convention: cache uses TARGET convention.
    tokens[t]  = base model's emitted token at step t
    hidden[t]  = base hidden state right before emitting tokens[t]
    head_k predicts tokens[t + k] from input at position t.

Equivalently:
    pred_k   = logits_k         if k == 0 else logits_k[:, :-k, :]
    target_k = tokens           if k == 0 else tokens[:, k:]

So head_0 reproduces the base model's own next-token prediction (and at
step 0, when lm_head is copied from base and ResBlocks are identity-init,
head_0_loss ≈ 0 on real cache). head_k>=1 looks k tokens further ahead.

Verified on 3 cache samples (`scripts/verify_cache_convention.py`):
    TARGET conv. argmax(hidden[t] @ W^T) == tokens[t]   : 81.2%
    INPUT  conv. argmax(hidden[t] @ W^T) == tokens[t+1] :  0.0%
    neither                                             : 18.8%
(Run 2026-05-11 on /scratch/li96/mz9869/cached_data_test/qwen25vl_long/.)

Padded positions are tagged with -100 (the standard CE ignore_index) by
collate_fn, so they do not contribute to the loss.

Total loss is a weighted sum: sum_k (w_k * loss_k). The default weights
[1.0, 0.8, 0.64] follow vanilla Medusa.

Returned dict:
    {
        "head_0_loss": float,
        "head_1_loss": float,
        ...,
        "total_loss": Tensor (scalar, with gradient),
    }
Per-head losses are floats (`.item()`) for logging only; total_loss
preserves its computation graph so .backward() works on it.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

IGNORE_INDEX = -100


def compute_loss(
    all_logits: List[Tensor],
    tokens: Tensor,
    weights: List[float],
) -> Dict[str, float | Tensor]:
    """Compute weighted multi-head CE loss with proper offsets.

    Args:
        all_logits: list of (B, L, V) tensors, one per head (k = 0..K-1).
        tokens:     (B, L) int64, padded with -100 (IGNORE_INDEX).
        weights:    list of `len(all_logits)` floats.

    Returns:
        dict with per-head scalars and a differentiable "total_loss".

    Raises:
        ValueError if length is too short for the deepest head, i.e.
        L < num_heads (head_{K-1} needs L >= K).
    """
    if len(all_logits) != len(weights):
        raise ValueError(
            f"#logits ({len(all_logits)}) != #weights ({len(weights)})"
        )
    if not all_logits:
        raise ValueError("compute_loss received empty all_logits")

    B, L = tokens.shape
    K = len(all_logits)
    if L < K:
        raise ValueError(
            f"sequence length L={L} is too short for {K} heads under TARGET "
            f"convention: head_{K - 1} needs L >= {K} (i.e. L > num_heads - 1). "
            "Increase max_length or seq_len_range."
        )

    losses: Dict[str, float | Tensor] = {}
    total_loss: Tensor | None = None

    for k, (logits_k, w_k) in enumerate(zip(all_logits, weights)):
        # TARGET convention: head_k predicts tokens[t + k] from position t.
        if k == 0:
            pred_k = logits_k.contiguous()
            target_k = tokens.contiguous()
        else:
            pred_k = logits_k[:, :-k, :].contiguous()
            target_k = tokens[:, k:].contiguous()

        loss_k = F.cross_entropy(
            pred_k.reshape(-1, pred_k.size(-1)),
            target_k.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        losses[f"head_{k}_loss"] = loss_k.detach().item()
        total_loss = w_k * loss_k if total_loss is None else total_loss + w_k * loss_k

    losses["total_loss"] = total_loss
    return losses
