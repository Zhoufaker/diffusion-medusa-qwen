"""Synthetic-logits acceptance tests for build_tree_folded_dynamic (Tier-1)."""
from __future__ import annotations

import random

import pytest
import torch

from decode.common import topk_masked
from decode.tree import (
    build_mask_and_positions,
    build_tree_folded,
    build_tree_folded_dynamic,
    per_depth_widths,
)

V = 64
ROOT = 999


def _logits_top(tokens_in_order, V=V):
    x = torch.full((V,), -20.0)
    for rank, tok in enumerate(tokens_in_order):
        x[tok] = 10.0 - 0.01 * rank
    return x.view(1, 1, V)


def _spec(fanout_cli):
    """CLI fanout [1, w2, ...] → dynamic cand_k = K-1 speculative widths."""
    return list(fanout_cli[1:])


def _sig(nodes):
    """Topology signature independent of flat_idx / pop order."""
    by = {n.flat_idx: n for n in nodes}

    def path(n):
        toks = []
        cur = n
        while True:
            toks.append(cur.token)
            if cur.parent == -1:
                break
            cur = by[cur.parent]
        return tuple(reversed(toks))

    return sorted(
        (n.depth, path(n), round(n.cum_logprob, 6), n.logprob)
        for n in nodes
    )


def _flat_layout(nodes):
    """Ordered flat presentation: (token, depth, parent_flat_idx) per flat_idx."""
    return [(n.token, n.depth, n.parent) for n in nodes]


def _mask_pos_sig(nodes, past_len=8, cont_base=8):
    mask, pos = build_mask_and_positions(
        nodes, past_len=past_len, cont_base=cont_base, dtype=torch.float16, device="cpu",
    )
    return mask.cpu().tolist(), pos.cpu().tolist()


def _assert_equiv(logits, fanout, max_nodes, depth1_floor=True):
    s = build_tree_folded(logits, ROOT, fanout, max_nodes, depth1_floor=depth1_floor)
    d = build_tree_folded_dynamic(
        logits, ROOT, _spec(fanout), max_nodes, depth1_floor=depth1_floor,
    )
    assert len(s) == len(d), (len(s), len(d), per_depth_widths(s, len(fanout)),
                              per_depth_widths(d, len(fanout)))
    assert _sig(s) == _sig(d), (_sig(s), _sig(d))
    assert _flat_layout(s) == _flat_layout(d), (_flat_layout(s), _flat_layout(d))
    assert _mask_pos_sig(s) == _mask_pos_sig(d)
    for a, b in zip(s, d):
        assert abs(a.cum_logprob - b.cum_logprob) < 1e-5, (a, b)


def test_topk_masked_is_log_softmax():
    """topk_masked returns finite log-softmax (lp <= 0), not raw logits."""
    x = torch.randn(128)
    lp, idx = topk_masked(x, 5)
    assert lp.isfinite().all()
    assert (lp <= 0).all()
    # Must match explicit log_softmax topk (phantom mask aside: V small here)
    from decode.common import mask_phantom_
    masked = mask_phantom_(x.clone())
    expect = torch.log_softmax(masked.float(), dim=-1).topk(5).values
    assert torch.allclose(lp.float(), expect)


def test_cand_k_rejects_len_eq_num_heads():
    logits = [_logits_top([0]), _logits_top([1, 2]), _logits_top([3, 4])]
    with pytest.raises(ValueError, match="speculative widths"):
        build_tree_folded_dynamic(logits, ROOT, [1, 2, 2], max_nodes=8)


def test_cli_to_depth2_width():
    """CLI --fanout 1 8 8 8 → cand_k[0]=8 realizes as depth-2 width under floor."""
    fanout = [1, 8, 8, 8]
    logits = [
        _logits_top([0]),
        _logits_top(list(range(10, 30))),
        _logits_top(list(range(30, 50))),
        _logits_top(list(range(50, 64))),
    ]
    d = build_tree_folded_dynamic(logits, ROOT, _spec(fanout), 32, depth1_floor=True)
    assert per_depth_widths(d, 4)[1] == 8


def test_flat_layout_byte_identical_gate2_shape():
    """Explicit layout assert: dynamic flat ≡ static flat element-wise (gate#2 shape)."""
    logits = [
        _logits_top([0]),
        _logits_top(list(range(10, 20))),
        _logits_top(list(range(20, 30))),
        _logits_top(list(range(30, 40))),
        _logits_top(list(range(40, 50))),
    ]
    fanout = [1, 6, 4, 2, 1]
    s = build_tree_folded(logits, ROOT, fanout, 24, depth1_floor=True)
    d = build_tree_folded_dynamic(logits, ROOT, _spec(fanout), 24, depth1_floor=True)
    assert _flat_layout(s) == _flat_layout(d)
    assert [n.flat_idx for n in s] == list(range(len(s)))
    assert [n.flat_idx for n in d] == list(range(len(d)))


def test_equiv_full_tree_fits():
    """gate#1-shaped: max_nodes >= fullN, floor no-op."""
    logits = [
        _logits_top([0]),
        _logits_top([10, 11, 12, 13]),
        _logits_top([20, 21, 22]),
        _logits_top([30, 31]),
    ]
    _assert_equiv(logits, [1, 3, 2, 1], max_nodes=16)


def test_equiv_budget_bind_with_floor_overshoot():
    """Budget binds + floor overshoot (gate#2 mechanism on synthetic logits)."""
    V = 128

    def L(scores):
        x = torch.full((V,), -100.0)
        for t, v in scores:
            x[t] = v
        return x.view(1, 1, V)

    logits = [
        L([(0, 0.0)]),
        L([(1, 20.0)] + [(2 + i, -8.0) for i in range(15)]),
        L([(40, 20.0)] + [(41 + i, 5.0) for i in range(5)]),
        L([(60, 20.0)] + [(61 + i, 5.0) for i in range(3)]),
    ]
    fanout = [1, 16, 4, 2]
    max_nodes = 10
    s = build_tree_folded(logits, ROOT, fanout, max_nodes, depth1_floor=True)
    d = build_tree_folded_dynamic(
        logits, ROOT, _spec(fanout), max_nodes, depth1_floor=True,
    )
    assert len(s) == len(d), (len(s), len(d))
    assert len(s) > max_nodes, f"floor must overshoot, got {len(s)}"
    assert per_depth_widths(s, 4)[1] == 16
    assert _sig(s) == _sig(d)
    assert _flat_layout(s) == _flat_layout(d)


def test_equiv_gate2_fanout_shape():
    """Same fanout/max_nodes as live gate#2; require static≡dynamic."""
    logits = [
        _logits_top([0]),
        _logits_top(list(range(10, 20))),
        _logits_top(list(range(20, 30))),
        _logits_top(list(range(30, 40))),
        _logits_top(list(range(40, 50))),
    ]
    _assert_equiv(logits, [1, 6, 4, 2, 1], max_nodes=24, depth1_floor=True)


def test_review_p1_exact_tie_siblings_flat_layout():
    """GPT-5.6 review P1 counterexample: exact-cum siblings must not permute.

    Speculative widths (2,2,2,4), max_nodes=7, floor off. Previously static
    set-iteration order could emit depth-5 as slot 1,0 while dynamic used 0,1.
    """
    V = 32

    def from_lp(lps_by_tok):
        # Invert log-softmax-ish: set raw logits so topk logprobs ≈ targets.
        # Simpler: build logits with equal values for tied tokens.
        x = torch.full((V,), -50.0)
        for tok, raw in lps_by_tok.items():
            x[tok] = raw
        return x.view(1, 1, V)

    # Depth-2: (0, -2) relative — use raw equal-gap logits
    # We need exact cum ties at depth-5. Construct via equal raw logits for
    # the two tied siblings so log_softmax gives identical lp.
    logits = [
        from_lp({0: 0.0}),
        from_lp({1: 0.0, 2: -2.0}),          # depth2 slots ~ (0,-2) after sm
        from_lp({3: 0.0, 4: -0.5}),
        from_lp({5: -0.5, 6: -1.75}),
        from_lp({7: 0.0, 8: 0.0, 9: -1.25, 10: -2.0}),  # exact tie 7 vs 8
    ]
    fanout = [1, 2, 2, 2, 4]
    _assert_equiv(logits, fanout, max_nodes=7, depth1_floor=False)


def test_tie_a_last_budget_slot_across_parents():
    V = 32

    def L(mapping):
        x = torch.full((V,), -50.0)
        for tok, val in mapping.items():
            x[tok] = val
        return x.view(1, 1, V)

    logits = [
        L({0: 0.0}),
        L({1: 5.0, 2: 5.0, 3: -10.0}),
        L({10: 3.0, 11: 3.0, 12: -10.0}),
    ]
    fanout = [1, 2, 2]
    _assert_equiv(logits, fanout, max_nodes=5, depth1_floor=True)
    _assert_equiv(logits, fanout, max_nodes=4, depth1_floor=False)


def test_tie_b_equal_cum_different_depths():
    V = 32

    def L(mapping):
        x = torch.full((V,), -50.0)
        for tok, val in mapping.items():
            x[tok] = val
        return x.view(1, 1, V)

    logits = [
        L({0: 0.0}),
        L({1: 4.0, 2: 0.0, 3: -20.0}),
        L({10: 0.0, 11: -1.0, 12: -20.0}),
    ]
    fanout = [1, 2, 2]
    for mn in (3, 4, 5, 6, 7):
        _assert_equiv(logits, fanout, max_nodes=mn, depth1_floor=True)
        _assert_equiv(logits, fanout, max_nodes=mn, depth1_floor=False)


def test_tie_c_floor_union_vs_heap_boundary():
    V = 64

    def L(tokens_scores):
        x = torch.full((V,), -40.0)
        for tok, sc in tokens_scores:
            x[tok] = sc
        return x.view(1, 1, V)

    logits = [
        L([(0, 0.0)]),
        L([(1, 10.0)] + [(i, -2.0 - 0.1 * i) for i in range(2, 10)]),
        L([(20, 10.0)] + [(20 + i, 0.0) for i in range(1, 6)]),
        L([(40, 10.0)] + [(40 + i, 0.0) for i in range(1, 4)]),
    ]
    fanout = [1, 8, 4, 2]
    s = build_tree_folded(logits, ROOT, fanout, max_nodes=12, depth1_floor=True)
    d = build_tree_folded_dynamic(
        logits, ROOT, _spec(fanout), max_nodes=12, depth1_floor=True,
    )
    assert len(s) == len(d)
    assert len(s) > 12, f"expected overshoot, got {len(s)}"
    assert per_depth_widths(s, 4)[1] == 8
    assert _sig(s) == _sig(d)
    assert _flat_layout(s) == _flat_layout(d)


def test_pop_order_must_not_be_rank():
    V = 32

    def L(mapping):
        x = torch.full((V,), -50.0)
        for tok, val in mapping.items():
            x[tok] = val
        return x.view(1, 1, V)

    logits = [
        L({0: 0.0}),
        L({1: 1.0, 2: 4.0, 3: -20.0}),
        L({10: 2.0, 11: 2.0, 12: -20.0}),
    ]
    fanout = [1, 2, 2]
    for mn in range(3, 8):
        _assert_equiv(logits, fanout, max_nodes=mn, depth1_floor=True)


def _quantize_fp16_grid(raw: torch.Tensor) -> torch.Tensor:
    """Quantize to fp16-representable values to induce exact ties."""
    return raw.half().float()


def test_property_based_static_dynamic_equiv_10k():
    """≥10k randomized cases: quantized logits, ties, budgets, floor on/off."""
    rng = random.Random(20260806)
    V = 48
    n_cases = 10_000
    for case in range(n_cases):
        # Folded: 3..5 heads → 2..4 speculative levels
        n_spec = rng.randint(2, 4)
        K = n_spec + 1
        widths = [rng.randint(1, 4) for _ in range(n_spec)]
        fanout = [1] + widths
        # Full product size (approx) to pick budget cuts
        full = 1
        prod = 1
        for w in widths:
            prod *= w
            full += prod
        # Budget cut points: below floor, at mid, above full
        choices = sorted({1, 2, max(1, full // 3), max(1, full // 2), full, full + 2})
        max_nodes = choices[rng.randrange(len(choices))]
        floor = rng.choice([True, False])

        logits = []
        # head 0 unused for topk in folded (forced root) but still present
        logits.append(torch.zeros(1, 1, V))
        for d in range(n_spec):
            # Mix strict order + exact sibling ties + cross-branch-ish duplicates
            raw = torch.randn(V, generator=torch.Generator().manual_seed(
                rng.randint(0, 2**31 - 1)
            ))
            if rng.random() < 0.4:
                # Force exact ties: duplicate top scores
                raw = _quantize_fp16_grid(raw)
                top = raw.topk(min(8, V)).indices
                if len(top) >= 2:
                    raw[top[1]] = raw[top[0]]
                if len(top) >= 4 and rng.random() < 0.5:
                    raw[top[3]] = raw[top[2]]
            else:
                raw = _quantize_fp16_grid(raw)
            logits.append(raw.view(1, 1, V))

        try:
            _assert_equiv(logits, fanout, max_nodes, depth1_floor=floor)
        except AssertionError as e:
            raise AssertionError(
                f"case={case} fanout={fanout} max_nodes={max_nodes} floor={floor}: {e}"
            ) from e
