"""Cosine LR schedule with linear warmup and a configurable floor.

Spec ref: linked_medusa_spec.md §3 — "Cosine decay to final_lr_multiplier=0.33 of peak".

We roll our own (rather than importing transformers' helper) because the
transformers import is slow and pulls in heavy modules we don't need.

Behaviour:
    step < warmup_steps          : lr_factor = step / warmup_steps  (linear ramp)
    warmup_steps <= step < total : lr_factor = floor + (1-floor) * 0.5 * (1 + cos(pi * progress))
    step >= total_steps          : lr_factor = floor

where `progress = (step - warmup_steps) / (total_steps - warmup_steps)`.
"""
from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_schedule(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    final_lr_multiplier: float = 0.33,
) -> LambdaLR:
    if warmup_steps < 0 or total_steps <= 0:
        raise ValueError(f"bad warmup={warmup_steps}, total={total_steps}")
    if not 0.0 <= final_lr_multiplier <= 1.0:
        raise ValueError(f"final_lr_multiplier must be in [0, 1]; got {final_lr_multiplier}")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        if step >= total_steps:
            return final_lr_multiplier
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_multiplier + (1.0 - final_lr_multiplier) * cosine

    return LambdaLR(optimizer, lr_lambda)
