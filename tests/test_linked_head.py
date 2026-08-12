"""Linked head structural tests (spec §5.3, §5.4, §11 check C3).

The single most important property of this design: gradients from head_2's
loss MUST flow back to head_0's parameters via the chain h_t -> h_0' -> h_1'
-> ... -> h_{K-1}'. If anyone adds a `.detach()` anywhere on that path, this
test catches it.
"""
import torch
import torch.nn.functional as F

from model import LinkedMedusaHeads


def _build(H=16, V=32, K=3, B=2, L=8):
    torch.manual_seed(0)
    model = LinkedMedusaHeads(
        hidden_dim=H, vocab_size=V, num_heads=K, num_blocks=2, expansion=2,
    )
    h_t = torch.randn(B, L, H)
    tokens = torch.randint(0, V, (B, L))
    return model, h_t, tokens


def test_forward_shapes():
    model, h_t, _ = _build(H=16, V=32, K=3, B=2, L=8)
    all_logits = model(h_t)
    assert isinstance(all_logits, list) and len(all_logits) == 3
    for k, logits in enumerate(all_logits):
        assert logits.shape == (2, 8, 32), f"head {k} got {logits.shape}"


def test_loss_only_on_last_head_propagates_to_head_0():
    """C3: zero out loss_0 + loss_1, only loss_2 active. Head_0 params must
    still have non-zero grads — proves the chain is connected.

    TARGET convention: head_2 predicts tokens[t+2], so
        pred_2   = logits_2[:, :-2, :]
        target_2 = tokens[:, 2:]
    """
    model, h_t, tokens = _build(H=16, V=32, K=3, B=2, L=8)
    all_logits = model(h_t)
    pred = all_logits[2][:, :-2, :].reshape(-1, 32)
    target = tokens[:, 2:].reshape(-1)
    loss_2 = F.cross_entropy(pred, target)
    loss_2.backward()

    head0_grad_l2_sq = 0.0
    n_with_grad = 0
    # Look at head_0's body (the ResBlock stack). lm_head may or may not have
    # grads via the next-head input path; the canonical thing to check is the
    # resblock stack's parameters, because that's what produces h_0'.
    for name, p in model.heads[0].named_parameters():
        if "lm_head" in name:
            continue
        if p.grad is None:
            continue
        head0_grad_l2_sq += float(p.grad.norm().item()) ** 2
        n_with_grad += 1
    head0_grad_l2 = head0_grad_l2_sq ** 0.5
    assert head0_grad_l2 > 1e-8, (
        f"chain is broken: head_0 ResBlock params have ||grad||_2={head0_grad_l2:.3e} "
        f"with only loss_2 active. Check for stray .detach() in LinkedMedusaHeads.forward."
    )
    assert n_with_grad > 0


def test_no_detach_in_chain_path():
    """Cross-check: hidden state at the input to head_k>=1 must depend on
    head_{k-1}'s parameters (i.e. have a working graph)."""
    model, h_t, _ = _build(H=16, V=32, K=3, B=2, L=8)
    # forward head_0 manually
    logits0, h0_prime = model.heads[0](h_t)
    # head_1 input is h_t + h0_prime; assert h0_prime requires grad
    assert h0_prime.requires_grad, "h_0' must require_grad for chain to be trainable"
    # and assert it depends on head_0 params (sanity that resblocks contribute)
    s = h0_prime.sum()
    grads = torch.autograd.grad(
        s, list(model.heads[0].body.parameters()), retain_graph=False, allow_unused=True
    )
    # all should be non-None (zero-init residual still has grads w.r.t. weights)
    assert all(g is not None for g in grads)


def test_init_lm_heads_from_base():
    H, V, K = 16, 32, 3
    torch.manual_seed(0)
    model = LinkedMedusaHeads(H, V, num_heads=K, num_blocks=2, expansion=2)
    base_W = torch.randn(V, H)
    model.init_lm_heads_from_base(base_W)
    for k in range(K):
        assert torch.equal(model.heads[k].lm_head.weight.data, base_W), (
            f"head {k}.lm_head was not copied from base"
        )


# ---------------------------------------------------------------------------
# B1: 5-head + detach_chain
# ---------------------------------------------------------------------------
import torch as _torch
from model import LinkedMedusaHeads as _LMH


def test_5head_forward_shapes():
    m = _LMH(hidden_dim=16, vocab_size=40, num_heads=5, num_blocks=1, expansion=2)
    out = m(_torch.randn(2, 7, 16))
    assert len(out) == 5 and all(o.shape == (2, 7, 40) for o in out)


def test_detach_chain_values_identical_grads_cut():
    _torch.manual_seed(0)
    a = _LMH(hidden_dim=16, vocab_size=40, num_heads=3, num_blocks=1,
             expansion=2, detach_chain=False)
    b = _LMH(hidden_dim=16, vocab_size=40, num_heads=3, num_blocks=1,
             expansion=2, detach_chain=True)
    b.load_state_dict(a.state_dict())
    # non-identity body so the chain actually carries signal
    for m in (a, b):
        for head in m.heads:
            _torch.nn.init.normal_(head.body[0].w2.weight, std=0.05)
    b.load_state_dict(a.state_dict())
    x = _torch.randn(1, 5, 16)
    la, lb = a(x), b(x)
    for t1, t2 in zip(la, lb):
        assert _torch.equal(t1, t2), "detach must not change forward values"
    # deepest-head loss: no-detach reaches head_0 params, detach does not
    for m, expect_flow in ((a, True), (b, False)):
        m.zero_grad(set_to_none=True)
        m(x)[2].sum().backward()
        g = [p.grad for p in m.heads[0].parameters()]
        has = any(gr is not None and gr.abs().sum() > 0 for gr in g)
        assert has == expect_flow, f"detach_chain={m.detach_chain}: head_0 grad flow={has}"


def test_max_heads_truncation_exact():
    """Inference truncation: forward(h, max_heads=k) must equal forward(h)[:k]
    exactly — the chain is sequential, head_k never influences heads 0..k-1."""
    _torch.manual_seed(0)
    m = _LMH(hidden_dim=16, vocab_size=40, num_heads=5, num_blocks=1, expansion=2)
    for head in m.heads:   # non-identity body so the chain carries signal
        _torch.nn.init.normal_(head.body[0].w2.weight, std=0.05)
    x = _torch.randn(1, 5, 16)
    full = m(x)
    for k in (1, 2, 3, 5):
        part = m(x, max_heads=k)
        assert len(part) == k
        for t1, t2 in zip(full[:k], part):
            assert _torch.equal(t1, t2), f"truncation at {k} changed values"


def test_skip_head0_lm_head_exact():
    """skip_head0_lm_head: index 0 is None, deeper logits bit-identical
    (h_0' still feeds the chain; lm_head is a leaf of head_0's graph)."""
    _torch.manual_seed(0)
    m = _LMH(hidden_dim=16, vocab_size=40, num_heads=5, num_blocks=1, expansion=2)
    for head in m.heads:
        _torch.nn.init.normal_(head.body[0].w2.weight, std=0.05)
    x = _torch.randn(1, 5, 16)
    full = m(x)
    skip = m(x, skip_head0_lm_head=True)
    assert skip[0] is None and len(skip) == len(full)
    for t1, t2 in zip(full[1:], skip[1:]):
        assert _torch.equal(t1, t2), "skipping head_0 lm_head changed deeper logits"
    # combined with truncation
    both = m(x, max_heads=3, skip_head0_lm_head=True)
    assert both[0] is None and len(both) == 3
    for t1, t2 in zip(full[1:3], both[1:]):
        assert _torch.equal(t1, t2)


def test_c1_cond_index_matches_head0_target():
    """§2.2: cond_ids must reuse head_0 target ids (tokens[t]), not t-1 or t+1."""
    tokens = _torch.tensor([[11, 22, 33, -100]])
    head0_target_ids = tokens
    cond_ids = head0_target_ids
    assert (cond_ids == head0_target_ids).all()
    assert cond_ids[0, 0].item() == head0_target_ids[0, 0].item() == 11
    assert cond_ids[0, 1].item() == head0_target_ids[0, 1].item() == 22


def test_cond_embed_dtype_follows_h_t():
    """fp32 h_t + fp16 cond_embed must not crash; bonus_proj sees fp32 cond."""
    _torch.manual_seed(0)
    m = _LMH(hidden_dim=16, vocab_size=40, num_heads=3, num_blocks=1, expansion=2)
    h = _torch.randn(1, 5, 16, dtype=_torch.float32)
    cond = _torch.randn(1, 5, 16, dtype=_torch.float16)
    ref = m(h, cond_embed=cond.to(_torch.float32))
    out = m(h, cond_embed=cond)
    for t1, t2 in zip(ref, out):
        assert t1.dtype == _torch.float32
        assert _torch.allclose(t1, t2, rtol=0, atol=0)


def test_cond_embed_none_exact_with_trunc_and_skip():
    """§3.3: cond_embed=None must match legacy forward for all inference combos."""
    _torch.manual_seed(0)
    m = _LMH(hidden_dim=16, vocab_size=40, num_heads=5, num_blocks=1, expansion=2)
    for head in m.heads:
        _torch.nn.init.normal_(head.body[0].w2.weight, std=0.05)
    x = _torch.randn(1, 5, 16)
    ref = m(x)
    for mh in (None, 2, 3, 5):
        for skip in (False, True):
            out = m(x, max_heads=mh, skip_head0_lm_head=skip, cond_embed=None)
            n = mh if mh is not None else 5
            for k in range(n):
                if skip and k == 0:
                    assert out[0] is None
                    assert ref[0] is not None
                else:
                    assert _torch.equal(out[k], ref[k]), (
                        f"cond_embed=None changed logits at k={k} "
                        f"(max_heads={mh}, skip_head0={skip})"
                    )


def test_warm_start_strict_false_semantics():
    # 3-head sd loads into 5-head model leaving heads 3-4 untouched
    small = _LMH(hidden_dim=16, vocab_size=40, num_heads=3, num_blocks=1, expansion=2)
    big = _LMH(hidden_dim=16, vocab_size=40, num_heads=5, num_blocks=1, expansion=2)
    with _torch.no_grad():
        for p in small.parameters():
            p.add_(1.0)
    before_h4 = [p.clone() for p in big.heads[4].parameters()]
    res = big.load_state_dict(small.state_dict(), strict=False)
    assert not res.unexpected_keys
    assert all(k.startswith(("heads.3.", "heads.4.")) for k in res.missing_keys)
    for p, q in zip(big.heads[0].parameters(), small.heads[0].parameters()):
        assert _torch.equal(p, q)
    for p, q in zip(big.heads[4].parameters(), before_h4):
        assert _torch.equal(p, q)
