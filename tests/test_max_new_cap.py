"""Acceptance tests for max_new hard-cap (round-2 review P1 soft-cap fix).

Groups:
  (A) max_new ∈ 1..K+1 truncation
  (B) EOS at every accepted-path depth
  (C) folded trailing-bonus room / no-room
Hard invariants: len(to_emit) ≤ remaining; EOS inclusive; no overshoot.
"""
from __future__ import annotations

import pytest

from decode.common import truncate_emit_path

EOS = 2
K = 5  # folded tree max accept depth


def test_remaining_zero_emits_nothing():
    out, hit = truncate_emit_path([1, 3, 4], 0, EOS)
    assert out == [] and hit is False


@pytest.mark.parametrize("max_new", list(range(1, K + 2)))
def test_group_a_max_new_1_through_k_plus_1(max_new):
    """Path longer than cap must truncate to exactly max_new (no EOS)."""
    path = list(range(100, 100 + K + 3))  # length K+3, no EOS
    out, hit = truncate_emit_path(path, max_new, EOS)
    assert hit is False
    assert len(out) == max_new
    assert out == path[:max_new]
    assert len(out) <= max_new


@pytest.mark.parametrize("eos_pos", list(range(K + 1)))
def test_group_b_eos_at_each_accepted_depth(eos_pos):
    """EOS at depth index eos_pos (0-based) stops inclusive; respects remaining."""
    path = [10 + i for i in range(K + 1)]
    path[eos_pos] = EOS
    # remaining large enough to reach EOS
    out, hit = truncate_emit_path(path, remaining=K + 1, eos_id=EOS)
    assert hit is True
    assert out == path[: eos_pos + 1]
    assert out[-1] == EOS
    assert len(out) <= K + 1

    # remaining cuts before EOS
    if eos_pos > 0:
        out2, hit2 = truncate_emit_path(path, remaining=eos_pos, eos_id=EOS)
        assert hit2 is False
        assert out2 == path[:eos_pos]
        assert EOS not in out2
        assert len(out2) <= eos_pos


def test_group_c_folded_trailing_bonus_with_room():
    """Trailing bonus (single token) emits when remaining≥1."""
    bonus = 42
    out, hit = truncate_emit_path([bonus], remaining=1, eos_id=EOS)
    assert out == [bonus] and hit is False
    out_eos, hit_eos = truncate_emit_path([EOS], remaining=1, eos_id=EOS)
    assert out_eos == [EOS] and hit_eos is True


def test_group_c_folded_trailing_bonus_no_room():
    out, hit = truncate_emit_path([42], remaining=0, eos_id=EOS)
    assert out == [] and hit is False


def test_nonfolded_path_accepted_plus_bonus_cap():
    """Non-folded emit path = accepted + bonus; cap mid-path skips reorg consumer."""
    accepted = [11, 12, 13]
    bonus = 99
    path = accepted + [bonus]
    # Cap inside accepted → no full path for continue
    out, hit = truncate_emit_path(path, remaining=2, eos_id=EOS)
    assert out == [11, 12] and hit is False
    assert len(out) < len(path)  # caller must skip reorg/bonus-fwd
    # Cap exactly at end of accepted (exclude bonus)
    out2, _ = truncate_emit_path(path, remaining=3, eos_id=EOS)
    assert out2 == accepted
    assert len(out2) < len(path)
    # Full path fits
    out3, _ = truncate_emit_path(path, remaining=4, eos_id=EOS)
    assert out3 == path


def test_simulate_multi_round_never_exceeds_max_new():
    """Toy multi-round loop mirrors runner: remaining each round, hard cap."""
    max_new = 7
    # Round paths that would overshoot under soft-cap (3+3+3 > 7); avoid EOS id
    rounds = [[11, 12, 13], [14, 15, 16], [17, 18, 19]]
    emitted = []
    for path in rounds:
        if len(emitted) >= max_new:
            break
        remaining = max_new - len(emitted)
        to_emit, hit = truncate_emit_path(path, remaining, EOS)
        emitted.extend(to_emit)
        if hit or len(emitted) >= max_new or len(to_emit) < len(path):
            break
    assert len(emitted) == max_new
    assert emitted == [11, 12, 13, 14, 15, 16, 17]
    assert len(emitted) <= max_new


def test_greedy_prefix_contract_shape():
    """Documented contract: emitted must equal greedy[:max_new] (prefix).

    Pure-logic stand-in: truncating a greedy sequence to max_new is idempotent
    and length-safe — the GPU e2e asserts the tree runner matches this prefix.
    """
    greedy = list(range(50, 50 + 20))
    for max_new in range(1, 12):
        expect = greedy[:max_new]
        out, _ = truncate_emit_path(greedy, max_new, EOS)
        assert out == expect
        assert len(out) <= max_new
