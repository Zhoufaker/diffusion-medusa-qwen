"""W2 unit tests (no GPU; design doc reports/w2_train_design.md).

Covers: mask correctness (3-block toy, §1c), dual-path logits equality
(sdpa vs eager on CPU; Flex logic-equivalence, numeric parity deferred to
smoke), anchor sampling range/uniformity (§1b), ctx alignment known-answer
incl off-by-one edges (§1d), loss weights (Eq.4), embed/lm_head freeze,
pilot-shard round-trip.
"""
import math
import random
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "external" / "dflash")):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.ctx_dataset import (  # noqa: E402
    IGNORE_INDEX, MASK_TOKEN_ID, additive_4d, allowed_bool_mask,
    block_weights, collate_packed, flex_mask_mod_factory, pack_blocks,
    sample_anchors,
)

PILOT = Path("/scratch/li96/mz9869/dflash_data/ctx_cache_35k_pilot")


# ---------------------------------------------------------------- §1c mask --
def _toy_mask():
    # design §1c figure: ctx len 6, B=3, anchors at p=2,3,4
    anchor_pos = torch.tensor([2, 3, 4])
    block_of = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return allowed_bool_mask(anchor_pos, block_of, 6, 9), anchor_pos, block_of


def test_mask_three_rules():
    allow, _, block_of = _toy_mask()
    # rule 3: ctx visibility strictly below anchor
    exp_ctx = torch.tensor([[1, 1, 0, 0, 0, 0]] * 3
                           + [[1, 1, 1, 0, 0, 0]] * 3
                           + [[1, 1, 1, 1, 0, 0]] * 3, dtype=torch.bool)
    assert torch.equal(allow[:, :6], exp_ctx)
    # rules 1+2: in-block full, cross-block zero — elementwise
    for q in range(9):
        for kv in range(9):
            assert allow[q, 6 + kv] == (block_of[q] == block_of[kv])


def test_flex_mask_mod_equals_bool_mask():
    allow, anchor_pos, block_of = _toy_mask()
    mm = flex_mask_mod_factory(anchor_pos[block_of], block_of, ctx_len=6)
    got = torch.zeros_like(allow)
    for q in range(9):
        for kv in range(15):
            got[q, kv] = mm(0, 0, torch.tensor(q), torch.tensor(kv))
    assert torch.equal(got, allow)


def test_dual_path_logits_identical_sdpa_vs_eager():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 9, 8)
    k = torch.randn(1, 2, 15, 8)
    v = torch.randn(1, 2, 15, 8)
    allow, _, _ = _toy_mask()
    am = additive_4d(allow, torch.float32)
    out_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=am)
    # eager reference
    scores = q @ k.transpose(-1, -2) / math.sqrt(8) + am
    out_eager = torch.softmax(scores, dim=-1) @ v
    assert torch.allclose(out_sdpa, out_eager, atol=1e-5)


def test_masked_positions_receive_zero_attention():
    allow, _, _ = _toy_mask()
    am = additive_4d(allow, torch.float32)
    scores = torch.zeros(1, 1, 9, 15) + am
    probs = torch.softmax(scores, dim=-1)
    assert torch.all(probs[0, 0][~allow] == 0)
    assert torch.allclose(probs.sum(-1), torch.ones(1, 1, 9))


# ------------------------------------------------------------ §1b anchors --
def test_anchor_range_and_uniformity():
    rng = random.Random(43)
    P, T = 379, 635  # pilot idx0 spans
    counts = torch.zeros(T)
    for _ in range(10_000):
        for p in sample_anchors(P, T, alpha=0.05, max_anchors=512, rng=rng):
            assert P <= p <= T - 2  # never vision/prompt, never label-less tail
            counts[p] += 1
    assert counts[:P].sum() == 0 and counts[T - 1] == 0
    dom = counts[P:T - 1]
    assert dom.min() > 0.5 * dom.float().mean()  # loose uniformity
    assert dom.max() < 2.0 * dom.float().mean()


def test_anchor_count_formula():
    rng = random.Random(0)
    P, T = 100, 130  # L=30
    assert len(sample_anchors(P, T, 1.5, 512, rng)) == min(45, T - 1 - P)
    assert len(sample_anchors(P, T, 2.0, 20, rng)) == 20  # max_anchors cap
    assert sample_anchors(10, 11, 1.0, 512, rng) == []    # L=1: no legal anchor


# ------------------------------------------------------- §1d ctx alignment --
def test_pack_blocks_known_answer_and_edges():
    T, B = 20, 4
    ids = torch.arange(100, 100 + T)
    # mid block, tail-partial block, last legal anchor
    packed = pack_blocks(ids, anchors=[5, 17, 18], block_size=B)
    # anchor slot: noise id = ids[p], no label
    assert packed.noise_ids[0] == 105 and packed.labels[0] == IGNORE_INDEX
    assert torch.equal(packed.labels[1:4], torch.tensor([106, 107, 108]))
    assert torch.equal(packed.noise_pos[:4], torch.tensor([5, 6, 7, 8]))
    assert (packed.noise_ids[1:4] == MASK_TOKEN_ID).all()
    # anchor 17: slots at 17,18,19,20 -> labels 118,119 then out-of-range
    assert torch.equal(packed.labels[4:8],
                       torch.tensor([IGNORE_INDEX, 118, 119, IGNORE_INDEX]))
    # anchor 18 = T-2 (last legal): exactly one label
    assert torch.equal(packed.labels[8:12],
                       torch.tensor([IGNORE_INDEX, 119, IGNORE_INDEX, IGNORE_INDEX]))


def test_ctx_alignment_strictness_off_by_one():
    # anchor p: ctx column p-1 visible, column p NOT visible (§1d sentinel)
    allow = allowed_bool_mask(torch.tensor([7]), torch.zeros(4, dtype=torch.long),
                              ctx_len_padded=10, n_noise=4)
    assert allow[0, 6] and not allow[0, 7]


# ------------------------------------------------------------- §1e weights --
def test_block_weights_formula_and_padding_zero():
    w = block_weights(16, 7.0)
    assert w[0] == 0.0
    for m in range(1, 16):
        assert torch.isclose(w[m], torch.exp(torch.tensor(-(m - 1) / 7.0)))
    from train.train_drafter import slot_weights_like
    labels = torch.full((1, 32), IGNORE_INDEX)
    labels[0, 1] = 5
    sw = slot_weights_like(labels, 16, 7.0)
    assert sw[0, 1] == 1.0 and sw.sum() == 1.0  # everything else zeroed


# ------------------------------------------------------------- freeze test --
def test_embed_lmhead_frozen_no_grad():
    from train.train_drafter import TrainConfig, build_drafter
    cfg = TrainConfig(hidden_size=64, intermediate_size=128,
                      num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, num_hidden_layers=2, vocab_size=256,
                      num_target_layers=8, block_size=4)
    model = build_drafter(cfg).float()
    embed = torch.nn.Embedding(256, 64).requires_grad_(False)
    lm_head = torch.nn.Linear(64, 256, bias=False).requires_grad_(False)
    ids = torch.arange(40, 60)
    packed = pack_blocks(ids, [8, 12], 4, mask_token_id=250)  # tiny-vocab mask id
    allow = allowed_bool_mask(packed.anchor_pos, packed.block_of, 20, 8)
    am = additive_4d(allow, torch.float32)
    pos = torch.cat([torch.arange(20), packed.noise_pos])[None]
    ctx = torch.randn(1, 20, len(model.target_layer_ids) * 64)
    out = model(position_ids=pos, attention_mask=am,
                noise_embedding=embed(packed.noise_ids)[None],
                target_hidden=ctx, is_causal=False)
    loss = lm_head(out).float().pow(2).mean()
    loss.backward()
    assert embed.weight.grad is None and lm_head.weight.grad is None
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads) and any(g.abs().sum() > 0 for g in grads)


# --------------------------------------------------------- pilot round-trip --
@pytest.mark.skipif(not PILOT.exists(), reason="pilot shards not accessible")
def test_pilot_shard_roundtrip_spans_and_collate():
    from data.ctx_dataset import CtxShardDataset
    ds = CtxShardDataset(str(PILOT), "ctx_manifest_pilot.json", indices=[0, 137])
    it0 = ds[0]
    vs, ve, ps, pe, rs, re = it0["spans"].tolist()
    assert (ps, pe) == (0, 379) and (rs, re) == (379, 635)  # manifest idx0
    assert it0["ctx"].shape == (5, 635, 3584) and it0["ids"].shape[0] == 635
    rng = random.Random(43)
    batch = collate_packed([ds[0], ds[1]], block_size=16, alpha=0.1,
                           max_anchors=8, rng=rng)
    Bs, N = batch["labels"].shape
    Tmax = batch["ctx_len"]
    assert batch["ctx"].shape[:2] == (2, 5) and batch["ctx"].shape[2] == Tmax
    assert batch["allow"].shape == (2, N, Tmax + N)
    # labels must reproduce ids at anchor+m positions
    for b, it in enumerate((ds[0], ds[1])):
        ids = it["ids"]
        lab = batch["labels"][b]
        posn = batch["noise_pos"][b]
        val = batch["valid_noise"][b]
        for s in range(N):
            if val[s] and lab[s] != IGNORE_INDEX:
                assert lab[s] == ids[posn[s]]
    # padded noise queries attend exactly key 0
    pad_rows = ~batch["valid_noise"]
    if pad_rows.any():
        sub = batch["allow"][pad_rows]
        assert torch.all(sub[:, 0]) and sub[:, 1:].sum() == 0
