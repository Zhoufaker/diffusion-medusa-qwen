"""CPU unit tests for decode.tree primitives (no model needed)."""
import torch

from decode.tree import (
    accept,
    build_mask_and_positions,
    build_tree,
    build_tree_folded,
    per_depth_widths,
    tree_tokens,
)

V = 32


def _logits_with_top(tokens_in_order):
    """A (1,1,V) logits tensor whose top-k argsort == tokens_in_order."""
    x = torch.full((V,), -10.0)
    for rank, tok in enumerate(tokens_in_order):
        x[tok] = 10.0 - rank  # strictly decreasing so topk order is deterministic
    return x.view(1, 1, V)


def _make_logits():
    # head0 top: [1,2,3,4]; head1 top: [5,6,7]; head2 top: [8,9]
    return [
        _logits_with_top([1, 2, 3, 4, 0]),
        _logits_with_top([5, 6, 7, 0]),
        _logits_with_top([8, 9, 0]),
    ]


def test_build_full_expansion_counts():
    nodes = build_tree(_make_logits(), fanout=[4, 3, 2], max_nodes=40, depth1_floor=True)
    w = per_depth_widths(nodes, 3)
    assert w == [4, 12, 24], w
    assert len(nodes) == 40


def test_prefix_closed_and_depths():
    nodes = build_tree(_make_logits(), fanout=[4, 3, 2], max_nodes=16, depth1_floor=True)
    by_idx = {n.flat_idx: n for n in nodes}
    for n in nodes:
        if n.parent == -1:
            assert n.depth == 1
        else:
            assert n.parent in by_idx, "prefix-closed violated"
            assert by_idx[n.parent].depth == n.depth - 1
    # depth-major flatten: all depth-1 indices precede depth-2, etc.
    depths = [n.depth for n in sorted(nodes, key=lambda x: x.flat_idx)]
    assert depths == sorted(depths)


def test_depth1_floor_keeps_all_roots():
    # tiny budget; floor must still keep all 4 depth-1 nodes
    nodes = build_tree(_make_logits(), fanout=[4, 3, 2], max_nodes=4, depth1_floor=True)
    w = per_depth_widths(nodes, 3)
    assert w[0] == 4, f"depth-1 floor failed: {w}"


def test_no_floor_can_drop_roots():
    nodes_floor = build_tree(_make_logits(), fanout=[4, 3, 2], max_nodes=2, depth1_floor=True)
    nodes_nofloor = build_tree(_make_logits(), fanout=[4, 3, 2], max_nodes=2, depth1_floor=False)
    assert per_depth_widths(nodes_floor, 3)[0] == 4
    assert per_depth_widths(nodes_nofloor, 3)[0] <= 2


def test_mask_and_positions():
    nodes = build_tree(_make_logits(), fanout=[4, 3, 2], max_nodes=40, depth1_floor=True)
    P, cont_base = 7, 5  # cont_base != P to mimic rope_delta offset
    mask, pos = build_mask_and_positions(nodes, P, cont_base, torch.float32, device="cpu")
    N = len(nodes)
    assert mask.shape == (1, 1, N, P + N)
    assert pos.shape == (1, N)
    mn = torch.finfo(torch.float32).min
    by_idx = {n.flat_idx: n for n in nodes}
    for n in nodes:
        i = n.flat_idx
        # all prefix visible
        assert (mask[0, 0, i, :P] == 0).all()
        # self visible
        assert mask[0, 0, i, P + i] == 0
        # position = cont_base + depth-1
        assert pos[0, i].item() == cont_base + (n.depth - 1)
        # ancestors visible, non-ancestors among tree nodes blocked
        anc = set()
        a = n.parent
        while a != -1:
            anc.add(a); a = by_idx[a].parent
        for j in range(N):
            if j == i or j in anc:
                assert mask[0, 0, i, P + j] == 0
            else:
                assert mask[0, 0, i, P + j] == mn


def test_accept_full_path():
    nodes = build_tree(_make_logits(), fanout=[4, 3, 2], max_nodes=40, depth1_floor=True)
    by = {n.flat_idx: n for n in nodes}
    # target path: token 1 (d1) -> token 5 (d2, child of the token-1 node) -> token 8 (d3)
    def find(depth, token, parent_flat):
        for n in nodes:
            if n.depth == depth and n.token == token and n.parent == parent_flat:
                return n.flat_idx
        raise AssertionError("node not found")
    n1 = find(1, 1, -1)
    n5 = find(2, 5, n1)
    n8 = find(3, 8, n5)
    N = len(nodes)
    v = torch.full((N, V), -10.0)
    # base argmax after node n1 -> 5 ; after n5 -> 8 ; after n8 -> 99-equivalent (token 13)
    v[n1, 5] = 50.0
    v[n5, 8] = 50.0
    v[n8, 13] = 50.0
    accepted, accept_len, bonus, depths = accept(nodes, v, base_pred_root=1)
    assert accepted == [n1, n5, n8], accepted
    assert accept_len == 3
    assert bonus == 13
    assert depths == [1, 2, 3]


def test_accept_stops_at_mismatch():
    nodes = build_tree(_make_logits(), fanout=[4, 3, 2], max_nodes=40, depth1_floor=True)
    N = len(nodes)
    v = torch.full((N, V), -10.0)
    # base_pred_root = 99 -> no depth-1 token matches -> accept_len 0, bonus stays 99
    accepted, accept_len, bonus, depths = accept(nodes, v, base_pred_root=99)
    assert accepted == []
    assert accept_len == 0
    assert bonus == 99


def test_folded_root_is_forced_known_next():
    logits = _make_logits()  # head1 top: [5,6,7]; head2 top: [8,9]
    KNOWN = 4242
    nodes = build_tree_folded(logits, KNOWN, fanout=[4, 3, 2], max_nodes=40)
    roots = [n for n in nodes if n.parent == -1]
    assert len(roots) == 1
    assert roots[0].token == KNOWN and roots[0].depth == 1
    # depth-2 from head1 (=[5,6,7]), depth-3 from head2 (=[8,9])
    w = per_depth_widths(nodes, 3)
    assert w == [1, 3, 6], w  # 1 root, 3 children, 3*2 grandchildren
    d2_tokens = sorted({n.token for n in nodes if n.depth == 2})
    assert d2_tokens == [5, 6, 7], d2_tokens
    d3_tokens = sorted({n.token for n in nodes if n.depth == 3})
    assert d3_tokens == [8, 9], d3_tokens


def test_folded_prefix_closed_and_root_kept_under_budget():
    nodes = build_tree_folded(_make_logits(), 4242, fanout=[4, 3, 2], max_nodes=3)
    by = {n.flat_idx: n for n in nodes}
    roots = [n for n in nodes if n.parent == -1]
    assert len(roots) == 1 and roots[0].token == 4242
    for n in nodes:
        if n.parent != -1:
            assert n.parent in by and by[n.parent].depth == n.depth - 1


def test_folded_accept_root_then_path():
    nodes = build_tree_folded(_make_logits(), 4242, fanout=[4, 3, 2], max_nodes=40)
    N = len(nodes)
    root = next(n for n in nodes if n.parent == -1)
    n5 = next(n for n in nodes if n.depth == 2 and n.token == 5)
    n8 = next(n for n in nodes if n.depth == 3 and n.token == 8 and n.parent == n5.flat_idx)
    v = torch.full((N, V), -10.0)
    v[root.flat_idx, 5] = 50.0     # base after root -> 5
    v[n5.flat_idx, 8] = 50.0       # base after 5 -> 8
    v[n8.flat_idx, 13] = 50.0      # base after 8 -> 13 (next known, in-range)
    accepted, accept_len, next_known, depths = accept(nodes, v, base_pred_root=4242)
    assert accepted[0] == root.flat_idx          # root always accepted
    assert [nodes[i].token for i in accepted] == [4242, 5, 8]
    assert next_known == 13
    assert depths == [1, 2, 3]


def test_tree_tokens_order():
    nodes = build_tree(_make_logits(), fanout=[2, 2, 2], max_nodes=14, depth1_floor=True)
    toks = tree_tokens(nodes, device="cpu")
    assert toks.shape == (1, len(nodes))
    assert toks[0].tolist() == [n.token for n in nodes]


# ---------------------------------------------------------------------------
# B1: 5-head folded (4 speculative levels) + first-spec-layer floor
# ---------------------------------------------------------------------------


def _make_logits_5():
    # head0 unused by folded; heads 1..4 top candidates
    return [
        _logits_with_top([1, 2, 3, 4, 0]),
        _logits_with_top([5, 6, 7, 12, 13, 14, 0]),   # depth-2 (fanout up to 6)
        _logits_with_top([8, 9, 10, 11, 0]),          # depth-3
        _logits_with_top([15, 16, 0]),                # depth-4
        _logits_with_top([17, 0]),                    # depth-5
    ]


def test_folded_5head_full_expansion_and_depths():
    # fanout [_,6,4,2,1]: full = 1 + 6 + 24 + 48 + 48 = 127 nodes
    nodes = build_tree_folded(_make_logits_5(), root_token=3,
                              fanout=[1, 6, 4, 2, 1], max_nodes=999)
    w = per_depth_widths(nodes, 5)
    assert w == [1, 6, 24, 48, 48], w
    assert nodes[0].token == 3 and nodes[0].depth == 1 and nodes[0].parent == -1


def test_folded_5head_prune_keeps_root_and_spec1_floor():
    nodes = build_tree_folded(_make_logits_5(), root_token=3,
                              fanout=[1, 6, 4, 2, 1], max_nodes=8, depth1_floor=True)
    w = per_depth_widths(nodes, 5)
    assert w[0] == 1, "forced root must survive"
    assert w[1] == 6, f"first speculative layer floor failed: {w}"
    # prefix closure
    by_idx = {n.flat_idx: n for n in nodes}
    for n in nodes:
        assert n.parent == -1 or n.parent in by_idx


def test_folded_5head_no_floor_prunes_spec1():
    nodes = build_tree_folded(_make_logits_5(), root_token=3,
                              fanout=[1, 6, 4, 2, 1], max_nodes=8, depth1_floor=False)
    w = per_depth_widths(nodes, 5)
    assert w[0] == 1
    assert len(nodes) >= 8  # top-N + prefix closure can exceed N slightly
    # without the floor, budget concentrates in the top-1 subtree
    assert w[1] < 6, f"expected spec-1 pruned without floor: {w}"


def test_folded_3head_default_tree_unchanged_by_floor():
    # Phase A config [_,3,2] max_nodes=16: full tree (1+3+6=10) fits budget, so
    # floor on/off must give identical trees (anchor numbers unaffected).
    logits = _make_logits()
    a = build_tree_folded(logits, 7, [1, 3, 2], 16, depth1_floor=True)
    b = build_tree_folded(logits, 7, [1, 3, 2], 16, depth1_floor=False)
    assert [(n.token, n.depth, n.parent) for n in a] == \
           [(n.token, n.depth, n.parent) for n in b]


def test_folded_5head_accept_deep_path():
    nodes = build_tree_folded(_make_logits_5(), root_token=3,
                              fanout=[1, 6, 4, 2, 1], max_nodes=64)
    # base agrees with path root(3) -> 5 -> 8 -> 15 -> 17 then bonus 21.
    # Same-token nodes exist under every parent at a level, so index the exact
    # path nodes by walking the parent chain (not by (depth, token) alone).
    def _child(parent_idx, token):
        return next(n.flat_idx for n in nodes
                    if n.parent == parent_idx and n.token == token)
    i1 = 0                       # forced root, token 3
    i2 = _child(i1, 5)
    i3 = _child(i2, 8)
    i4 = _child(i3, 15)
    i5 = _child(i4, 17)
    v = torch.full((len(nodes), V), -10.0)
    v[i1, 5] = 10.0
    v[i2, 8] = 10.0
    v[i3, 15] = 10.0
    v[i4, 17] = 10.0
    v[i5, 21] = 10.0
    acc, alen, bonus, depths = accept(nodes, v, base_pred_root=3)
    assert alen == 5 and depths == [1, 2, 3, 4, 5] and bonus == 21
