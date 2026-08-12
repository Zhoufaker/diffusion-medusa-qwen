"""Loss offset & ignore_index tests (spec §6.4).

These guard the most error-prone part of the whole codebase: the per-head
token offset. We test against a hand-built fixture where we know exactly
which tokens each head should be predicting.
"""
import math

import torch
import torch.nn.functional as F

from train.loss import compute_loss, IGNORE_INDEX


def test_loss_returns_per_head_keys_and_total():
    B, L, V, K = 2, 6, 5, 3
    torch.manual_seed(0)
    all_logits = [torch.randn(B, L, V, requires_grad=True) for _ in range(K)]
    tokens = torch.randint(0, V, (B, L))
    weights = [1.0, 0.8, 0.64]
    out = compute_loss(all_logits, tokens, weights=weights)
    for k in range(K):
        assert f"head_{k}_loss" in out
    assert "total_loss" in out
    # total_loss should be a tensor (gradient-bearing); per-head are floats
    assert torch.is_tensor(out["total_loss"])
    assert not torch.is_tensor(out["head_0_loss"])


def test_loss_offset_matches_manual_ce():
    """TARGET convention: head_k predicts tokens[t+k] from input at position t.

        k == 0:  pred = logits_0,                 target = tokens
        k >= 1:  pred = logits_k[:, :-k, :],      target = tokens[:, k:]

    Verified against cache via scripts/verify_cache_convention.py.
    """
    B, L, V = 1, 8, 4
    torch.manual_seed(0)
    logits = [torch.randn(B, L, V) for _ in range(3)]
    tokens = torch.arange(L).unsqueeze(0).clamp_(max=V - 1)
    out = compute_loss(logits, tokens, weights=[1.0, 1.0, 1.0])
    for k in range(3):
        if k == 0:
            pred = logits[k].reshape(-1, V)
            tgt = tokens.reshape(-1)
        else:
            pred = logits[k][:, :-k, :].reshape(-1, V)
            tgt = tokens[:, k:].reshape(-1)
        manual = F.cross_entropy(pred, tgt, ignore_index=IGNORE_INDEX).item()
        assert abs(out[f"head_{k}_loss"] - manual) < 1e-6, (
            f"head_{k}_loss = {out[f'head_{k}_loss']}, manual = {manual}"
        )


def test_loss_total_is_weighted_sum():
    B, L, V = 1, 6, 4
    torch.manual_seed(0)
    logits = [torch.randn(B, L, V) for _ in range(3)]
    tokens = torch.randint(0, V, (B, L))
    weights = [1.0, 0.8, 0.64]
    out = compute_loss(logits, tokens, weights=weights)
    expected = sum(w * out[f"head_{k}_loss"] for k, w in enumerate(weights))
    assert abs(out["total_loss"].item() - expected) < 1e-5


def test_loss_ignores_pad_positions():
    """Positions marked with IGNORE_INDEX must not contribute."""
    B, L, V = 1, 6, 4
    torch.manual_seed(0)
    logits = [torch.randn(B, L, V) for _ in range(3)]
    # all targets are -100 -> CE loss should be 0
    tokens = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    out = compute_loss(logits, tokens, weights=[1.0, 1.0, 1.0])
    for k in range(3):
        # F.cross_entropy returns nan when all targets are ignored; we accept
        # either 0.0 (which would happen if implementation falls back) or nan.
        v = out[f"head_{k}_loss"]
        assert v == 0.0 or math.isnan(v), f"head_{k}_loss = {v}"


def test_loss_short_sequence_raises():
    """Under TARGET convention, head_{K-1} needs L >= K (i.e. L > num_heads - 1).

    With num_heads=3 and L=2, head_2 needs L >= 3, so loss code must raise."""
    B, L, V = 1, 2, 4  # L=2 < num_heads=3 -> head_2 needs L>=3, must raise
    logits = [torch.randn(B, L, V) for _ in range(3)]
    tokens = torch.randint(0, V, (B, L))
    try:
        compute_loss(logits, tokens, weights=[1.0, 1.0, 1.0])
    except (ValueError, RuntimeError, AssertionError):
        return
    raise AssertionError("compute_loss should raise on L < num_heads")


def test_loss_boundary_length_passes():
    """L == num_heads should be the *minimum* legal length under TARGET conv.

    With num_heads=3 and L=3, head_2 needs L>=3, exactly satisfied:
      pred_2   = logits_2[:, :-2, :]  -> (B, 1, V)
      target_2 = tokens[:, 2:]        -> (B, 1)
    Should compute a valid scalar loss."""
    B, L, V = 1, 3, 4
    torch.manual_seed(0)
    logits = [torch.randn(B, L, V) for _ in range(3)]
    tokens = torch.randint(0, V, (B, L))
    out = compute_loss(logits, tokens, weights=[1.0, 0.8, 0.64])
    for k in range(3):
        assert isinstance(out[f"head_{k}_loss"], float)
    assert torch.is_tensor(out["total_loss"])
