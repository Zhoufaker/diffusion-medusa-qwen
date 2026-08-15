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
    batch = collate_packed([ds[0], ds[1]], block_size=16, alpha=0.1,
                           max_anchors=8, seed=43, epoch=0)
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


# ------------------------------------------------- F1: short-sample filter --
def _fake_cache(tmp_path, Ls=(1, 2, 150), n_layers=5, H=8):
    from safetensors.torch import save_file
    tensors, samples = {}, {}
    for i, L in enumerate(Ls):
        P = 10
        T = P + L
        ids = torch.arange(T)
        tensors[f"{i}.ids"] = ids
        tensors[f"{i}.ctx"] = torch.zeros(n_layers, T, H, dtype=torch.bfloat16)
        tensors[f"{i}.spans"] = torch.tensor([2, 6, 0, P, P, T])
        samples[str(i)] = {"idx": i, "T": T, "P": P, "L": L, "n_pos": L,
                           "n_match": L - (1 if L > 2 else 0),
                           "mismatch_head16": ([{"pos": L - 1, "gap": 0.1}]
                                               if L > 2 else [])}
    save_file(tensors, str(tmp_path / "shard_00000.safetensors"))
    import json as _j
    _j.dump({"shard_size": 256, "samples": samples},
            open(tmp_path / "ctx_manifest.json", "w"))
    return tmp_path


def test_short_sample_filter_and_no_crash(tmp_path):
    from data.ctx_dataset import CtxShardDataset
    d = _fake_cache(tmp_path)
    ds = CtxShardDataset(str(d))
    assert ds.filtered_short == [0]          # L=1 dropped, counted
    assert ds.indices == [1, 2]              # L=2 kept (1 legal anchor)
    batch = collate_packed([ds[0], ds[1]], block_size=4, alpha=2.0,
                           max_anchors=8, seed=43, epoch=0)
    assert batch["valid_noise"].any()
    # collate on an L<2 item (filter bypassed) must trip the K=0 assert
    bad = {"ids": torch.arange(11), "ctx": torch.zeros(5, 11, 8),
           "spans": torch.tensor([0, 0, 0, 10, 10, 11]), "meta": {"idx": 99}}
    with pytest.raises(AssertionError, match="K=0"):
        collate_packed([bad], 4, 2.0, 8, seed=43, epoch=0)


# ------------------------------------------------------- F2: val split ------
def test_val_split_deterministic_disjoint_500():
    from train.train_drafter import make_val_split
    all_idx = list(range(0, 34999, 7))
    t1, v1 = make_val_split(all_idx, 43, 500)
    t2, v2 = make_val_split(all_idx, 43, 500)
    assert v1 == v2 and t1 == t2             # deterministic
    assert len(v1) == 500
    assert set(t1).isdisjoint(v1)
    assert sorted(t1 + v1) == all_idx        # partition, train hard-excludes val


# ---------------------------------------------- F3: anchor reproducibility --
def _syn_item(idx, P=20, T=60):
    return {"ids": torch.arange(T), "ctx": torch.zeros(5, T, 8),
            "spans": torch.tensor([2, 6, 0, P, P, T]), "meta": {"idx": idx}}


def test_anchor_rng_derivation():
    kw = dict(block_size=4, alpha=1.0, max_anchors=16)
    b_e0a = collate_packed([_syn_item(7)], seed=43, epoch=0, **kw)
    b_e0b = collate_packed([_syn_item(7)], seed=43, epoch=0, **kw)
    b_e1 = collate_packed([_syn_item(7)], seed=43, epoch=1, **kw)
    b_other = collate_packed([_syn_item(8)], seed=43, epoch=0, **kw)
    assert torch.equal(b_e0a["noise_pos"], b_e0b["noise_pos"])   # re-entrant
    assert not torch.equal(b_e0a["noise_pos"], b_e1["noise_pos"])  # epoch moves
    assert not torch.equal(b_e0a["noise_pos"], b_other["noise_pos"])  # idx moves
    # val convention: epoch pinned to 0 -> identical across "epochs" by constr.
    assert torch.equal(
        collate_packed([_syn_item(7)], seed=43, epoch=0, **kw)["noise_pos"],
        b_e0a["noise_pos"])


# ------------------------------------------- F5/门4: load fidelity (CPU) ----
def test_eval_only_load_fidelity_cpu(tmp_path):
    from train.train_drafter import (TrainConfig, build_drafter, save_drafter,
                                     slot_weights_like, weighted_ce)
    cfg = TrainConfig(hidden_size=64, intermediate_size=128,
                      num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, num_hidden_layers=2, vocab_size=256,
                      num_target_layers=8, block_size=4)
    torch.manual_seed(1)
    model = build_drafter(cfg).float()
    # quantize once through the checkpoint dtype so save->load is lossless
    model.load_state_dict({k: v.to(torch.bfloat16).float()
                           for k, v in model.state_dict().items()})
    lm_head = torch.nn.Linear(64, 256, bias=False).requires_grad_(False)
    ids = torch.arange(40, 60)
    packed = pack_blocks(ids, [8, 12], 4, mask_token_id=250)
    allow = allowed_bool_mask(packed.anchor_pos, packed.block_of, 20, 8)
    am = additive_4d(allow, torch.float32)
    pos = torch.cat([torch.arange(20), packed.noise_pos])[None]
    ctx = torch.randn(1, 20, len(model.target_layer_ids) * 64)

    def val_ce(m):
        with torch.no_grad():
            out = m(position_ids=pos, attention_mask=am,
                    noise_embedding=torch.randn(0).new_zeros(1, 8, 64) + 0.1,
                    target_hidden=ctx, is_causal=False)
            logits = lm_head(out)
        labels = packed.labels[None].clamp(max=255)
        w = slot_weights_like(labels, 4, 7.0)
        return float(weighted_ce(logits, labels, w))

    ce_before = val_ce(model)
    save_drafter(model, cfg, tmp_path, "best", step=1)
    ck = torch.load(tmp_path / "drafter_best.pt", map_location="cpu",
                    weights_only=True)
    model2 = build_drafter(cfg).float()
    model2.load_state_dict(ck["state_dict"])
    ce_after = val_ce(model2)
    assert ce_before == ce_after             # bitwise identical on CPU


# ------------------------------------- resume bundle equivalence (任务 2) ----
def test_resume_bundle_two_epoch_equivalence(tmp_path):
    """1 epoch + bundle-resume + 1 epoch == 2 continuous epochs, per-step
    loss trajectories bitwise identical (CPU fp32 deterministic; tolerance 0).
    Exercises: bundle round-trip (model/optimizer/scheduler/RNG), (seed,epoch)-
    derived shuffle + anchors."""
    import functools
    from train.train_drafter import (TrainConfig, build_drafter, run_epoch,
                                     save_bundle, load_bundle, epoch_generator)
    from data.ctx_dataset import CtxShardDataset
    d = _fake_cache(tmp_path, Ls=(20, 33, 47, 61, 28, 39, 55, 44),
                    n_layers=2, H=64)
    cfg = TrainConfig(hidden_size=64, intermediate_size=128,
                      num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, num_hidden_layers=2, vocab_size=512,
                      num_target_layers=8, block_size=4, batch_seqs=2,
                      lr=1e-3, epochs=2)

    def fresh():
        torch.manual_seed(7)
        m = build_drafter(cfg).float()
        e = torch.nn.Embedding(512, 64).requires_grad_(False)
        h = torch.nn.Linear(64, 512, bias=False).requires_grad_(False)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        return m, e, h, opt, sch

    def loader(epoch):
        ds = CtxShardDataset(str(d))
        collate = functools.partial(
            collate_packed, block_size=4, alpha=1.0, max_anchors=8,
            seed=cfg.data_seed, epoch=epoch, mask_token_id=500)
        return torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_seqs, shuffle=True,
            generator=epoch_generator(cfg.data_seed, epoch),
            collate_fn=collate, num_workers=0)

    dev = torch.device("cpu")
    # A: continuous 2 epochs
    m, e, h, opt, sch = fresh()
    lossesA = []
    for ep in range(2):
        lossesA += run_epoch(m, e, h, loader(ep), cfg, dev, opt, sch)["step_losses"]
    # B: 1 epoch -> bundle -> fresh objects -> resume -> 1 epoch
    m, e, h, opt, sch = fresh()
    lossesB = run_epoch(m, e, h, loader(0), cfg, dev, opt, sch)["step_losses"]
    save_bundle(m, opt, sch, epoch_done=0, gstep=4, best=min(lossesB), out=tmp_path)
    m2, e2, h2, opt2, sch2 = fresh()
    ep_done, gstep, best = load_bundle(tmp_path / "bundle_latest.pt",
                                       m2, opt2, sch2, dev)
    assert ep_done == 0 and gstep == 4
    lossesB += run_epoch(m2, e2, h2, loader(ep_done + 1), cfg, dev, opt2, sch2)["step_losses"]
    assert lossesA == lossesB                # bitwise, tolerance 0
    # single-copy invariant: exactly one bundle file, no tmp remnants
    assert [p.name for p in tmp_path.glob("bundle_latest.pt*")] == ["bundle_latest.pt"]
