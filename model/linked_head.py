"""LinkedMedusaHead and LinkedMedusaHeads.

Spec ref: linked_medusa_spec.md §5.2 / §5.3.

Architecture:
    Each head consumes a hidden state of shape (B, L, H), runs:
        input_resblock -> N body resblocks -> lm_head
    and returns BOTH the logits and the pre-lm_head hidden state h'.

    The full module chains 3 heads:
        head_0 input  = h_t
        head_k input  = h_t + h_{k-1}'        (for k >= 1, elementwise add)

    No .detach() anywhere — gradients from loss_2 must reach head_0
    parameters through the residual hidden-state path. This is the core
    novelty vs vanilla Medusa.

vocab_size note (per §2 of the updated spec):
    Qwen2.5-VL-7B-Instruct's lm_head has physical output dim 152064 (padded
    for 64/128-divisible hardware alignment). The effective tokenizer vocab
    is 151936; the extra 128 rows are never emitted by the tokenizer but are
    kept so the lm_head shape matches the base model's lm_head exactly.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn

from .resblock import MLPResBlock


class LinkedMedusaHead(nn.Module):
    """Single linked draft head. Returns (logits, pre-lm_head hidden)."""

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_blocks: int = 2,
        expansion: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.expansion = expansion

        self.input_resblock = MLPResBlock(hidden_dim, expansion)
        self.body = nn.Sequential(
            *[MLPResBlock(hidden_dim, expansion) for _ in range(num_blocks)]
        )
        # No bias on lm_head: matches the base model's lm_head, which we
        # copy in via init_lm_heads_from_base. Base Qwen2.5 lm_head has no
        # bias term either.
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: Tensor, skip_lm_head: bool = False) -> Tuple[Tensor | None, Tensor]:
        """x: (B, L, H) -> (logits: (B, L, V) or None, h_prime: (B, L, H)).

        skip_lm_head: compute only h' (resblocks), returning logits=None.
        Inference optimization for consumers that need this head's hidden for
        the chain but never read its logits (e.g. head_0 in the folded tree,
        whose depth-1 root is forced). h' is unaffected — lm_head is a leaf."""
        h = self.input_resblock(x)
        h = self.body(h)
        logits = None if skip_lm_head else self.lm_head(h)
        return logits, h


class LinkedMedusaHeads(nn.Module):
    """Three (configurable) linked draft heads chained via hidden-state passing."""

    def __init__(
        self,
        hidden_dim: int = 3584,
        vocab_size: int = 152064,
        num_heads: int = 3,
        num_blocks: int = 2,
        expansion: int = 2,
        detach_chain: bool = False,
    ):
        """detach_chain: if True, stop-grad the inter-head hidden h_{k-1}' before
        feeding head_k (forward values unchanged; gradients no longer flow through
        the chain). Fallback for deep-chain (5-head) training instability — the
        default False preserves the linked premise. No parameters are affected,
        so checkpoints are interchangeable across both settings."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.expansion = expansion
        self.detach_chain = detach_chain

        self.heads = nn.ModuleList(
            [
                LinkedMedusaHead(hidden_dim, vocab_size, num_blocks, expansion)
                for _ in range(num_heads)
            ]
        )
        self.bonus_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.bonus_proj.weight)

    def forward(self, h_t: Tensor, max_heads: int | None = None,
                cond_embed: Tensor | None = None,
                skip_head0_lm_head: bool = False) -> List[Tensor]:
        """h_t: (B, L, H) -> list of tensors of shape (B, L, V).

        head_0 input: h_t
        head_k input (k >= 1): h_t + h_{k-1}'
        head_1 input (k == 1, cond_embed given): h_t + h_{k-1}' + bonus_proj(cond_embed)

        h_{k-1}' is the pre-lm_head hidden of the previous head. By default
        NO .detach(): gradients must flow from the deepest loss back to head_0
        parameters (the linked premise). With detach_chain=True the chain
        VALUES are identical but gradients stop at each head boundary.

        max_heads: run only the first `max_heads` heads (inference optimization
        for trees that don't use deeper levels). The chain is strictly sequential
        (head_k depends only on heads 0..k-1), so truncation is EXACT:
        forward(h)[:k] == forward(h, max_heads=k). Training must pass None
        (all heads' losses are needed).

        cond_embed: optional (B, L, H) embedding vectors (NOT token ids).
        When None, all heads use the legacy input expressions. When set, only
        head_1 adds bonus_proj(cond_embed) to its input; k==0 and k>=2 are
        unchanged. Callers supply embed(bonus_token) from the frozen base model.

        skip_head0_lm_head: skip head_0's lm_head GEMV (folded inference only).
        """
        active = self.heads if max_heads is None else self.heads[:max_heads]
        all_logits: List[Tensor] = []
        h_prev_prime: Tensor | None = None
        for k, head in enumerate(active):
            prev = h_prev_prime
            if prev is not None and self.detach_chain:
                prev = prev.detach()
            head_input = h_t if prev is None else h_t + prev
            if k == 1 and cond_embed is not None:
                head_input = head_input + self.bonus_proj(cond_embed.to(h_t.dtype))
            logits, h_prime = head(head_input,
                                   skip_lm_head=(k == 0 and skip_head0_lm_head))
            all_logits.append(logits)
            h_prev_prime = h_prime
        return all_logits

    @torch.no_grad()
    def init_lm_heads_from_base(self, base_lm_head_weight: Tensor) -> None:
        """Copy a single (V, H) tensor into every head's lm_head.weight.

        Call this AFTER instantiation, BEFORE training. Per Medusa convention
        (and confirmed in spec §2), this gives near-zero head_0 CE loss at
        init because the body resblocks are identity-init, so:
            head_0 forward(h_t) ≈ base_lm_head @ h_t == base model's next-token prediction
        """
        if base_lm_head_weight.shape != (self.vocab_size, self.hidden_dim):
            raise ValueError(
                f"base lm_head weight has shape {tuple(base_lm_head_weight.shape)}, "
                f"expected {(self.vocab_size, self.hidden_dim)}"
            )
        for head in self.heads:
            head.lm_head.weight.data.copy_(base_lm_head_weight)
