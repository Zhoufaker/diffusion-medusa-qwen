"""Verify MLPResBlock is identity at init (spec §3, §5.1).

Why this matters: the design relies on the residual branch being literally
zero at step 0 so that head_0_lm_head ≈ base_lm_head(h_t) gives near-zero
CE on real cache (spec §7.1 sanity check). If anyone "fixes" the init by
removing zero-init, this test catches it before training.
"""
import torch

from model import MLPResBlock


def test_resblock_is_identity_at_init():
    torch.manual_seed(0)
    block = MLPResBlock(hidden_dim=64, expansion=2)
    x = torch.randn(2, 7, 64)
    y = block(x)
    assert torch.equal(x, y), (
        f"MLPResBlock must be EXACTLY identity at init (the residual branch's "
        f"final linear is zero-init'd). Got max abs diff = "
        f"{(x - y).abs().max().item()}"
    )


def test_resblock_w2_is_zero_init():
    block = MLPResBlock(hidden_dim=64, expansion=2)
    assert torch.all(block.w2.weight == 0), "w2.weight should be zero-initialized"
    assert torch.all(block.w2.bias == 0), "w2.bias should be zero-initialized"


def test_resblock_breaks_identity_after_a_step():
    """After one optimizer step the block is no longer identity. Sanity that
    the zero-init is a *starting point*, not a permanent constraint."""
    torch.manual_seed(0)
    block = MLPResBlock(hidden_dim=16, expansion=2)
    opt = torch.optim.SGD(block.parameters(), lr=1e-2)
    x = torch.randn(3, 5, 16, requires_grad=False)
    target = torch.randn(3, 5, 16)
    y = block(x)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    opt.step()
    y2 = block(x)
    assert not torch.equal(x, y2), "block should diverge from identity after a step"
