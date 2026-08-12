"""decode.tree — tree speculative decoding primitives.

Layering (per review Q5):
  REUSABLE (inference + P1 on-policy training):
      TreeNode, build_tree, build_mask_and_positions, tree_tokens,
      reorg_kv_safe, reorg_kv_gather, per_depth_widths
  INFERENCE-ONLY (greedy argmax matching; P1 will NOT use this — it scores
  against the base's actually-emitted tokens with a loss, not argmax equality):
      accept

Conventions (locked by scripts/probe_4d_mask.py):
  - Linked heads produce K FIXED per-level distributions from one h_t; the tree
    is the product of per-level top-f_k candidates (design doc §1).
  - Mask: (1,1,N,P+N) additive fp16; 0.0 attend, finfo.min block. Each node
    attends prefix + ancestor chain + self.
  - position_ids: node at tree-depth d -> cont_base + (d-1), where
    cont_base = past_len + base.model.rope_deltas (decode.common.continuation_base).
    M-RoPE offset is MANDATORY on image prompts.
  - KV reorg needs NO RoPE re-rotation: an accepted depth-d node had
    position_id cont_base+(d-1), exactly its sequential slot after reorg, so
    its already-rotated key is valid in place (design doc §5.2).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch

from .common import argmax_masked, topk_masked


@dataclass
class TreeNode:
    token: int          # candidate token id
    depth: int          # 1-based tree depth (1 = first draft position)
    parent: int         # flat_idx of parent, or -1 for depth-1 (root = prefix)
    logprob: float      # log p_{depth-1}(token) from that head's softmax
    cum_logprob: float  # sum of logprobs along root->this path (joint path logprob)
    flat_idx: int       # own index in the flattened node list


# ----------------------------------------------------------------------------
# Build + prune (REUSABLE)
# ----------------------------------------------------------------------------


def build_tree(
    all_logits: Sequence[torch.Tensor],
    fanout: Sequence[int],
    max_nodes: int,
    depth1_floor: bool = True,
) -> List[TreeNode]:
    """Build and prune a candidate tree from per-head logits.

    all_logits : list of K tensors, each (1,1,V) or (V,), one per head.
    fanout     : per-head top-k widths, len == K (e.g. [4,3,2]).
    max_nodes  : prune target N (keep top-N by joint cum_logprob).
    depth1_floor: force-keep ALL depth-1 nodes (review Q1). Rationale: heads are
                 overconfident, joint-prob top-N piles budget into the top-1
                 deep subtree and would starve depth-1 #2..#4 — exactly the
                 layer that most needs widening (35% of rounds die at depth-1).

    Returns nodes flattened in (depth, parent, -cum) order with remapped parents.
    Prefix-closed by construction (parent.cum >= child.cum, plus floor adds only
    parentless roots).
    """
    K = len(fanout)
    assert len(all_logits) == K, f"fanout len {K} != num head logits {len(all_logits)}"

    # per-level top-f_k (shared across all parents at that level — heads give a
    # single fixed distribution per level from h_t).
    levels: List[Tuple[List[float], List[int]]] = []
    for k in range(K):
        lg = all_logits[k].reshape(-1)
        lp, idx = topk_masked(lg, fanout[k])
        assert lp.isfinite().all() and (lp <= 0).all(), (
            "topk_masked must return finite log-softmax (lp<=0); got "
            f"min={float(lp.min())} max={float(lp.max())}"
        )
        levels.append((lp.tolist(), idx.tolist()))

    nodes: List[dict] = []
    # depth 1 (constructive rank = slot; same formula as folded dynamic)
    prev_layer: List[int] = []
    lp0, idx0 = levels[0]
    for a in range(fanout[0]):
        nodes.append(dict(token=idx0[a], depth=1, parent=-1, logprob=lp0[a],
                          cum=lp0[a], rank=a))
        prev_layer.append(len(nodes) - 1)
    # deeper levels
    for k in range(1, K):
        lp_k, idx_k = levels[k]
        width = fanout[k]
        cur_layer: List[int] = []
        for parent_tmp in prev_layer:
            pcum = nodes[parent_tmp]["cum"]
            p_rank = nodes[parent_tmp]["rank"]
            for b in range(width):
                nodes.append(dict(
                    token=idx_k[b], depth=k + 1, parent=parent_tmp,
                    logprob=lp_k[b], cum=pcum + lp_k[b],
                    rank=p_rank * width + b,
                ))
                cur_layer.append(len(nodes) - 1)
        prev_layer = cur_layer

    # prune: top-N by cum, then depth-1 floor, then prefix-close (defensive)
    order = sorted(range(len(nodes)), key=lambda i: nodes[i]["cum"], reverse=True)
    keep = set(order[:max_nodes])
    if depth1_floor:
        keep.update(i for i in range(len(nodes)) if nodes[i]["depth"] == 1)
    closed = set(keep)
    for i in list(keep):
        p = nodes[i]["parent"]
        while p != -1 and p not in closed:
            closed.add(p)
            p = nodes[p]["parent"]
    keep = closed

    # re-flatten: depth-major; final key = constructive rank so exact-cum ties
    # are deterministic (set iteration order is NOT a valid tie-break).
    kept_sorted = sorted(
        keep,
        key=lambda i: (
            nodes[i]["depth"], nodes[i]["parent"], -nodes[i]["cum"], nodes[i]["rank"],
        ),
    )
    remap = {tmp: new for new, tmp in enumerate(kept_sorted)}
    out: List[TreeNode] = []
    for new, tmp in enumerate(kept_sorted):
        nd = nodes[tmp]
        parent_new = -1 if nd["parent"] == -1 else remap[nd["parent"]]
        out.append(TreeNode(token=nd["token"], depth=nd["depth"], parent=parent_new,
                            logprob=nd["logprob"], cum_logprob=nd["cum"], flat_idx=new))
    return out


def build_tree_folded(
    all_logits: Sequence[torch.Tensor],
    root_token: int,
    fanout: Sequence[int],
    max_nodes: int,
    depth1_floor: bool = True,
) -> List[TreeNode]:
    """Fold-bonus tree: the known bonus becomes the FORCED depth-1 root.

    Eliminates the separate per-round bonus forward (design doc §0.4). The tree
    is drafted from the LAST-ACCEPTED node's hidden; its depth-1 token is the
    bonus (base's known argmax there), so head_0 is "spent" re-predicting the
    known bonus and the speculative layers come from head_1..head_{K-1}:
        depth-1 : [root_token]                       (forced, always kept/accepted)
        depth-2 : all_logits[1] top-fanout[1]        (children of root)
        depth-d : all_logits[d-1] top-fanout[d-1]    (children of depth-(d-1))
    i.e. one fewer speculative layer than build_tree (the cost of folding).
    Generalizes to any K (B1: K=5 -> 4 speculative levels from heads 1..4).
    Prune to max_nodes by joint cum_logprob (root has cum=0, always survives).

    depth1_floor: force-keep ALL depth-2 nodes — the FIRST SPECULATIVE layer,
    the folded analogue of build_tree's depth-1 floor (same rationale: heads
    are overconfident; global top-N piles budget into the top-1 deep subtree
    and starves the widest-accept layer). No-op when the full tree fits in
    max_nodes (e.g. the 3-head [.,3,2] tree of 10 nodes; Phase A folded
    numbers are therefore unaffected by this flag).
    """
    K = len(fanout)
    assert len(all_logits) == K
    # root (rank=0); speculative levels use heads 1..K-1
    nodes: List[dict] = [
        dict(token=int(root_token), depth=1, parent=-1, logprob=0.0, cum=0.0, rank=0)
    ]
    prev_layer = [0]
    for k in range(1, K):
        width = fanout[k]
        lp_t, idx_t = topk_masked(all_logits[k].reshape(-1), width)
        assert lp_t.isfinite().all() and (lp_t <= 0).all(), (
            "topk_masked must return finite log-softmax (lp<=0); got "
            f"min={float(lp_t.min())} max={float(lp_t.max())}"
        )
        lp_k, idx_k = lp_t.tolist(), idx_t.tolist()
        cur_layer: List[int] = []
        for parent_tmp in prev_layer:
            pcum = nodes[parent_tmp]["cum"]
            p_rank = nodes[parent_tmp]["rank"]
            for b in range(width):
                nodes.append(dict(
                    token=idx_k[b], depth=k + 1, parent=parent_tmp,
                    logprob=lp_k[b], cum=pcum + lp_k[b],
                    rank=p_rank * width + b,
                ))
                cur_layer.append(len(nodes) - 1)
        prev_layer = cur_layer

    order = sorted(range(len(nodes)), key=lambda i: nodes[i]["cum"], reverse=True)
    keep = set(order[:max(1, max_nodes)])
    keep.add(0)  # root floor
    if depth1_floor:
        keep.update(i for i in range(len(nodes)) if nodes[i]["depth"] == 2)
    closed = set(keep)
    for i in list(keep):
        p = nodes[i]["parent"]
        while p != -1 and p not in closed:
            closed.add(p); p = nodes[p]["parent"]
    keep = closed
    # Final key = constructive rank: set(keep) iteration is NOT a valid
    # tie-break under exact cum ties (GPT-5.6 review P1).
    kept_sorted = sorted(
        keep,
        key=lambda i: (
            nodes[i]["depth"], nodes[i]["parent"], -nodes[i]["cum"], nodes[i]["rank"],
        ),
    )
    remap = {tmp: new for new, tmp in enumerate(kept_sorted)}
    out: List[TreeNode] = []
    for new, tmp in enumerate(kept_sorted):
        nd = nodes[tmp]
        parent_new = -1 if nd["parent"] == -1 else remap[nd["parent"]]
        out.append(TreeNode(token=nd["token"], depth=nd["depth"], parent=parent_new,
                            logprob=nd["logprob"], cum_logprob=nd["cum"], flat_idx=new))
    return out


def build_tree_folded_dynamic(
    all_logits: Sequence[torch.Tensor],
    root_token: int,
    cand_k: Sequence[int],
    max_nodes: int,
    depth1_floor: bool = True,
) -> List[TreeNode]:
    """Best-first folded tree (docs/dynamic_tree_design.md B2/B3).

    ``cand_k`` MUST be exactly ``K-1`` speculative widths for depths 2..K
    (index ``cand_k[d-2]``). CLI ``--fanout 1 w2 w3 ...`` must pass
    ``fanout[1:]`` at the call site. When those widths equal static
    ``fanout[1:]``, node-/layout-identical to ``build_tree_folded`` under the
    tie-deterministic reflatten keys (any ``max_nodes``; post-hoc depth-2 floor).
    Does NOT modify ``build_tree_folded``.
    """
    K = len(all_logits)
    n_spec = K - 1
    if len(cand_k) == K:
        raise ValueError(
            f"cand_k must be exactly {n_spec} speculative widths (depths 2..{K}); "
            f"got len={len(cand_k)} (==num_heads, unused head-0 slot). "
            f"Pass fanout[1:] from CLI --fanout."
        )
    assert len(cand_k) == n_spec, (
        f"cand_k len {len(cand_k)} != K-1={n_spec} (K=len(all_logits)={K})"
    )
    # Precompute shared per-depth top-k. level_cand[d] = (lp, tok, slot_b).
    # Batched .tolist() once per depth — no per-element CUDA scalar sync.
    level_cand: Dict[int, List[Tuple[float, int, int]]] = {}
    for d in range(2, K + 1):
        width = cand_k[d - 2]
        if width <= 0:
            level_cand[d] = []
            continue
        lp_t, idx_t = topk_masked(all_logits[d - 1].reshape(-1), width)
        assert lp_t.isfinite().all() and (lp_t <= 0).all(), (
            "topk_masked must return finite log-softmax (lp<=0); got "
            f"min={float(lp_t.min())} max={float(lp_t.max())}"
        )
        lp_list, idx_list = lp_t.tolist(), idx_t.tolist()
        level_cand[d] = [(lp_list[b], idx_list[b], b) for b in range(width)]

    # Selected nodes; rank = parent.rank * cand_k[depth-2] + slot_b
    root = dict(token=int(root_token), depth=1, parent=-1, logprob=0.0, cum=0.0,
                rank=0, slot_b=-1)
    selected: List[dict] = [root]
    selected_keys = {(1, int(root_token), -1)}
    budget = max(1, max_nodes)

    heap: List[Tuple] = []
    seq = 0

    def push_children(parent_sel_idx: int) -> None:
        nonlocal seq
        p = selected[parent_sel_idx]
        d_child = p["depth"] + 1
        if d_child > K or not level_cand.get(d_child):
            return
        width = cand_k[d_child - 2]
        for lp, tok, b in level_cand[d_child]:
            child = dict(
                token=tok, depth=d_child, parent=parent_sel_idx,
                logprob=lp, cum=p["cum"] + lp,
                rank=p["rank"] * width + b, slot_b=b,
            )
            seq += 1
            heapq.heappush(
                heap,
                (-child["cum"], child["depth"], p["rank"], b, seq, child),
            )

    push_children(0)

    while len(selected) < budget and heap:
        _neg_cum, _dep, _prank, _b, _seq, child = heapq.heappop(heap)
        pidx = child["parent"]
        key = (child["depth"], child["token"], pidx)
        if key in selected_keys:
            continue
        selected_keys.add(key)
        new_idx = len(selected)
        selected.append(child)
        # Do not expand children once the pre-floor budget is full.
        if len(selected) < budget:
            push_children(new_idx)

    # Post-hoc floor union: add missing depth-2 (overshoot; not charged to budget)
    if depth1_floor and level_cand.get(2):
        have_d2 = {(nd["token"], nd["slot_b"]) for nd in selected if nd["depth"] == 2}
        width2 = cand_k[0]
        for lp, tok, b in level_cand[2]:
            if (tok, b) in have_d2:
                continue
            child = dict(
                token=tok, depth=2, parent=0, logprob=lp, cum=0.0 + lp,
                rank=0 * width2 + b, slot_b=b,
            )
            selected.append(child)

    # Reflatten (presentation): (depth, constructive rank) — unchanged.
    tmp_nodes = selected
    order = sorted(
        range(len(tmp_nodes)),
        key=lambda i: (tmp_nodes[i]["depth"], tmp_nodes[i]["rank"]),
    )
    remap = {tmp: new for new, tmp in enumerate(order)}
    out: List[TreeNode] = []
    for new, tmp in enumerate(order):
        nd = tmp_nodes[tmp]
        parent_new = -1 if nd["parent"] == -1 else remap[nd["parent"]]
        out.append(TreeNode(
            token=nd["token"], depth=nd["depth"], parent=parent_new,
            logprob=nd["logprob"], cum_logprob=nd["cum"], flat_idx=new,
        ))
    return out


def tree_tokens(nodes: List[TreeNode], device: str = "cuda:0") -> torch.Tensor:
    """(1, N) long tensor of candidate tokens in flat order."""
    return torch.tensor([[nd.token for nd in nodes]], device=device, dtype=torch.long)


def per_depth_widths(nodes: List[TreeNode], num_heads: int) -> List[int]:
    """Count of kept nodes at each depth (review Q1 logging)."""
    w = [0] * num_heads
    for nd in nodes:
        if 1 <= nd.depth <= num_heads:
            w[nd.depth - 1] += 1
    return w


# ----------------------------------------------------------------------------
# Mask + positions (REUSABLE)
# ----------------------------------------------------------------------------


def build_mask_and_positions(
    nodes: List[TreeNode],
    past_len: int,
    cont_base: int,
    dtype: torch.dtype,
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """4D tree attention mask (1,1,N,P+N) + 2D position_ids (1,N).

    cont_base MUST be decode.common.continuation_base(base, past_len)
    (i.e. past_len + rope_delta), not raw past_len.
    """
    N = len(nodes)
    P = past_len
    mn = torch.finfo(dtype).min
    mask = torch.full((1, 1, N, P + N), mn, dtype=dtype, device=device)
    mask[0, 0, :, :P] = 0.0
    pos = torch.empty(N, dtype=torch.long, device=device)
    for nd in nodes:
        i = nd.flat_idx
        mask[0, 0, i, P + i] = 0.0           # self
        a = nd.parent
        while a != -1:                       # ancestor chain
            mask[0, 0, i, P + a] = 0.0
            a = nodes[a].parent
        pos[i] = cont_base + (nd.depth - 1)
    return mask, pos.unsqueeze(0)


# ----------------------------------------------------------------------------
# Accept (INFERENCE-ONLY: greedy argmax matching)
# ----------------------------------------------------------------------------


def accept(
    nodes: List[TreeNode],
    v_logits: torch.Tensor,
    base_pred_root: int,
    argmax_fn: Callable[[torch.Tensor], int] = argmax_masked,
) -> Tuple[List[int], int, int, List[int]]:
    """Greedy tree acceptance.

    v_logits : (N, V) — base's next-token distribution AFTER each node's path.
    base_pred_root : base's argmax for the FIRST draft position (carried from the
                     previous round's bonus forward / prefill).
    Returns (accepted_flat_idx[in depth order], accept_len, bonus_token,
             accepted_depths). Walk the single greedy path: at each level pick
             the child whose token == base's argmax there. Bonus = base's argmax
             after the last accepted node (always a "free" correct token).
    """
    children: Dict[int, List[TreeNode]] = {}
    for nd in nodes:
        children.setdefault(nd.parent, []).append(nd)

    accepted: List[int] = []
    accepted_depths: List[int] = []
    expected = base_pred_root
    cur_parent = -1
    while True:
        match = next((nd for nd in children.get(cur_parent, []) if nd.token == expected), None)
        if match is None:
            break
        accepted.append(match.flat_idx)
        accepted_depths.append(match.depth)
        expected = argmax_fn(v_logits[match.flat_idx])
        cur_parent = match.flat_idx
        if cur_parent not in children:        # reached a leaf of the pruned tree
            break
    return accepted, len(accepted), expected, accepted_depths


# ----------------------------------------------------------------------------
# KV reorganization (REUSABLE)
# ----------------------------------------------------------------------------


def reorg_kv_safe(
    base,
    past_kv,
    past_len: int,
    accepted_tokens: List[int],
    cont_base: int,
    device: str = "cuda:0",
) -> None:
    """Safe reorg (recompute). Crop tree KV back to prefix, then re-run the
    accepted path linearly with correct M-RoPE positions. ~0 risk; costs one
    forward of accept_len(<=K) tokens. Recommended for v1 (review Q2)."""
    past_kv.crop(past_len)
    if accepted_tokens:
        ids = torch.tensor([accepted_tokens], device=device, dtype=torch.long)
        pos = torch.arange(cont_base, cont_base + len(accepted_tokens),
                           device=device).unsqueeze(0)
        base(input_ids=ids, past_key_values=past_kv, position_ids=pos, use_cache=True)


def reorg_kv_gather(
    past_kv,
    past_len: int,
    accepted_flat_idx: List[int],
    device: str = "cuda:0",
) -> None:
    """Gather reorg (no recompute). Keep prefix + accepted nodes' K/V by
    index_select on the cache seq dim. Validate against reorg_kv_safe
    (bit-identical next-round state) before trusting (review Q2). No RoPE
    re-rotation needed (design doc §5.2)."""
    if not accepted_flat_idx:
        past_kv.crop(past_len)
        return
    keep = torch.cat([
        torch.arange(past_len, device=device, dtype=torch.long),
        past_len + torch.tensor(accepted_flat_idx, device=device, dtype=torch.long),
    ])
    for layer in past_kv.layers:
        layer.keys = layer.keys.index_select(2, keep).contiguous()
        layer.values = layer.values.index_select(2, keep).contiguous()
