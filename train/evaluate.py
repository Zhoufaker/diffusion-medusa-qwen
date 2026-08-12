"""evaluate.py — per-head eval metrics.

Spec ref: linked_medusa_spec.md §6.5.

For each head k, on a val subset:
    - mean_loss : per-head CE loss (excluding -100 positions)
    - top1_acc  : fraction of positions where argmax(logits_k) == target_k
    - top5_acc  : fraction where target_k is in top-5 of logits_k

Padded positions (target == -100) are excluded from the denominator.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .loss import IGNORE_INDEX


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    loss_weights: List[float],
    use_amp: bool = False,
    max_batches: int | None = None,
    embed_layer: torch.nn.Module | None = None,
) -> Dict[str, float]:
    """Run evaluation over val_loader (or first max_batches batches)."""
    model.eval()
    num_heads = len(loss_weights)
    sum_loss = [0.0] * num_heads
    sum_top1 = [0] * num_heads
    sum_top5 = [0] * num_heads
    sum_valid = [0] * num_heads

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else torch.amp.autocast(device.type, enabled=False)
    )

    for b_idx, batch in enumerate(val_loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        h_t = batch["hidden"].to(device, non_blocking=True)
        if use_amp and device.type == "cuda":
            h_t = h_t.half()
        else:
            h_t = h_t.float()
        tokens = batch["tokens"].to(device, non_blocking=True)

        with autocast_ctx:
            if embed_layer is not None:
                head0_target_ids = tokens
                cond_ids = head0_target_ids
                assert (cond_ids == head0_target_ids).all(), (
                    "C1 cond index regression: cond_ids must equal head_0 target ids"
                )
                embed_ids = cond_ids.detach().cpu().clamp(min=0)
                cond_embed = embed_layer(embed_ids)
                if (cond_ids == IGNORE_INDEX).any():
                    cond_embed = cond_embed.masked_fill(
                        (cond_ids.detach().cpu() == IGNORE_INDEX).unsqueeze(-1), 0.0
                    )
                cond_embed = cond_embed.to(device, non_blocking=True)
                all_logits = model(h_t, cond_embed=cond_embed)
            else:
                all_logits = model(h_t)

        for k, logits_k in enumerate(all_logits):
            # TARGET convention (see train/loss.py docstring): head_k predicts
            # tokens[t + k] from position t. head_0 is full-length; head_k>=1
            # drops the last k positions of pred and the first k of target.
            if k == 0:
                pred_k = logits_k.float()             # cast for numerical stability
                target_k = tokens
            else:
                pred_k = logits_k[:, :-k, :].float()
                target_k = tokens[:, k:]
            mask = target_k != IGNORE_INDEX
            valid_n = int(mask.sum().item())
            if valid_n == 0:
                continue
            # CE loss on valid positions only
            loss = F.cross_entropy(
                pred_k.reshape(-1, pred_k.size(-1)),
                target_k.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            sum_loss[k] += float(loss.item())

            # top-1 / top-5
            top5_pred = pred_k.topk(5, dim=-1).indices                       # (B, L-offset, 5)
            target_exp = target_k.unsqueeze(-1)                              # (B, L-offset, 1)
            top1_hits = (top5_pred[..., :1] == target_exp).any(dim=-1) & mask
            top5_hits = (top5_pred == target_exp).any(dim=-1) & mask
            sum_top1[k] += int(top1_hits.sum().item())
            sum_top5[k] += int(top5_hits.sum().item())
            sum_valid[k] += valid_n

    metrics: Dict[str, float] = {}
    weighted_total_loss = 0.0
    for k in range(num_heads):
        n = max(1, sum_valid[k])
        mean_loss_k = sum_loss[k] / n
        metrics[f"eval/head_{k}_loss"] = mean_loss_k
        metrics[f"eval/head_{k}_top1"] = sum_top1[k] / n
        metrics[f"eval/head_{k}_top5"] = sum_top5[k] / n
        weighted_total_loss += loss_weights[k] * mean_loss_k
    metrics["eval/total_loss"] = weighted_total_loss
    return metrics
