"""MLPResBlock — pre-LN MLP residual block, identity-initialized.

Spec ref: linked_medusa_spec.md §5.1.

forward: x -> x + W_2(SiLU(W_1(LayerNorm(x))))

Init scheme:
    - W_1, b_1, LayerNorm: PyTorch defaults (Kaiming-uniform for Linear,
      ones/zeros for LN).
    - W_2.weight, W_2.bias: ZERO. Block is therefore exactly identity at init.

Identity-init lets us copy the base model's lm_head into each head.lm_head
and still have head_0 forward equal to (base lm_head applied to h_t),
which gives near-zero head_0 CE loss at step 0 — a reliable sanity check.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class MLPResBlock(nn.Module):
    """Pre-LN residual MLP block. Output ≡ x at initialization."""

    def __init__(self, hidden_dim: int, expansion: int = 2):
        super().__init__()
        inner_dim = hidden_dim * expansion
        self.norm = nn.LayerNorm(hidden_dim)
        self.w1 = nn.Linear(hidden_dim, inner_dim, bias=True)
        self.act = nn.SiLU()
        self.w2 = nn.Linear(inner_dim, hidden_dim, bias=True)

        # Identity-init: zero out the last linear so the residual branch
        # contributes 0 at step 0. The normalized + W_1 + SiLU path can be
        # arbitrary; what matters is W_2(.) == 0.
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(self, x: Tensor) -> Tensor:  # (B, L, H) -> (B, L, H)
        return x + self.w2(self.act(self.w1(self.norm(x))))
