"""D2 单测(CPU):适配器循环小例(mock target/drafter 全流程)、D3 埋点
schema、Latin square 覆盖均衡、paired bootstrap 正确性(合成数据)。"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "external" / "dflash"), str(_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from d2_eval_harness import latin_order, paired_bootstrap_ci  # noqa: E402
from decode.dflash_vlm import dflash_vlm_generate, run_twice_check  # noqa: E402

V, H, B = 64, 8, 4


class MockCache:
    def __init__(self): self.n = 0
    def get_seq_length(self): return self.n
    def crop(self, k): self.n = min(self.n, k)


class MockTarget:
    """Scripted target: argmax at global position g == ans[g+1] (one-hot)."""
    def __init__(self, ans, P):
        self.ans, self.P = ans, P
        self.device = torch.device("cpu")
        self.cache = MockCache()

    def __call__(self, input_ids=None, past_key_values=None, use_cache=True,
                 output_hidden_states=False, **kw):
        if past_key_values is None:
            self.cache = MockCache()      # prefill == fresh cache (真模型语义)
        if input_ids is None:
            input_ids = kw["input_ids"]
        T = input_ids.shape[1]
        g0 = self.cache.n
        logits = torch.full((1, T, V), -10.0)
        for t in range(T):
            nxt = self.ans[g0 + t + 1] if g0 + t + 1 < len(self.ans) else 0
            logits[0, t, nxt] = 10.0
        hs = tuple(torch.zeros(1, T, H) for _ in range(9))  # 8 layers + embed
        self.cache.n = g0 + T
        return SimpleNamespace(logits=logits, hidden_states=hs,
                               past_key_values=self.cache)


class MockDrafter(torch.nn.Module):
    """perfect=True: slot m-1 one-hot 命中 ans[start+m];否则输出全零。"""
    def __init__(self, ans, perfect):
        super().__init__()
        self.ans, self.perfect = ans, perfect
        self.target_layer_ids = [1, 2, 3, 4, 5]   # max tuple idx 6 < 9
        self.block_size = B

    def forward(self, target_hidden=None, noise_embedding=None,
                position_ids=None, past_key_values=None, use_cache=False,
                is_causal=False):
        pos = position_ids[0, -B:]
        hid = torch.zeros(1, B, V)
        if self.perfect:
            # mask 自去噪约定:槽 m 的输出行 m 预测位置 p+m 自身的 token
            for m in range(1, B):
                g = int(pos[m])
                if g < len(self.ans):
                    hid[0, m, self.ans[g]] = 5.0
        return hid


def _setup(perfect, P=6, n_new=9):
    ans = list(range(1, P + 1)) + [10 + i for i in range(n_new)] + [2] * B
    tgt = MockTarget(ans, P)
    dr = MockDrafter(ans, perfect)
    embed = torch.nn.Embedding(200, V)
    lm_head = torch.nn.Identity()
    inputs = {"input_ids": torch.tensor([ans[:P]])}
    return tgt, dr, embed, lm_head, inputs, ans


def test_adapter_perfect_drafter_full_accept():
    tgt, dr, embed, lm_head, inputs, ans = _setup(perfect=True)
    d3 = []
    seq, stats = dflash_vlm_generate(tgt, dr, embed, lm_head, inputs,
                                     max_new_tokens=8, eos_id=-1,
                                     block_size=B, mask_token_id=199,
                                     d3_log=d3)
    P = 6
    assert seq == ans[P: P + len(seq)]           # 输出跟脚本序列一致
    assert all(a == B - 1 for a in stats["accept_lengths"][:-1])  # 全接受
    assert stats["tau_accept_only"] <= B - 1
    assert stats["tau_with_bonus"] == stats["tau_accept_only"] + 1
    # D3 schema
    r = d3[0]
    assert set(r) == {"cycle", "anchor_pos", "slot_top1_prob", "slot_accept",
                      "accept_len", "reject_verify_argmax"}
    assert len(r["slot_top1_prob"]) == B - 1 and len(r["slot_accept"]) == B - 1


def test_adapter_zero_drafter_advances_one_per_cycle():
    tgt, dr, embed, lm_head, inputs, ans = _setup(perfect=False)
    seq, stats = dflash_vlm_generate(tgt, dr, embed, lm_head, inputs,
                                     max_new_tokens=6, eos_id=-1,
                                     block_size=B, mask_token_id=199)
    assert seq == ans[6: 6 + len(seq)]           # verify 兜底,输出仍正确
    assert all(a == 0 for a in stats["accept_lengths"])   # 每轮纯 bonus 推进
    assert stats["tau_with_bonus"] == 1.0


def test_adapter_crop_bookkeeping():
    tgt, dr, embed, lm_head, inputs, ans = _setup(perfect=True)
    dflash_vlm_generate(tgt, dr, embed, lm_head, inputs, max_new_tokens=8,
                        eos_id=-1, block_size=B, mask_token_id=199)
    # target cache 每轮 crop 到已接受长度:终态 == P + 已产出(<= max_len)
    assert tgt.cache.n <= 6 + 8 + B


def test_g2_hook_identical_on_deterministic_mock():
    tgt_args = _setup(perfect=True)
    r = run_twice_check(MockTarget(tgt_args[5], 6), tgt_args[1], tgt_args[2],
                        tgt_args[3], tgt_args[4], max_new_tokens=8, eos_id=-1,
                        block_size=B, mask_token_id=199)
    assert r["identical"] and r["first_div"] is None


def test_latin_square_balance():
    from collections import Counter
    slot_arm = [Counter() for _ in range(3)]
    for i in range(300):
        o = latin_order(i)
        assert sorted(o) == [0, 1, 2]
        for s, a in enumerate(o):
            slot_arm[s][a] += 1
    for s in range(3):
        assert all(slot_arm[s][a] == 100 for a in range(3))


def test_paired_bootstrap_ci():
    import random
    rng = random.Random(0)
    pos = [0.5 + rng.gauss(0, 0.1) for _ in range(300)]
    point, lo, hi = paired_bootstrap_ci(pos, n_boot=2000, seed=43)
    assert lo > 0 and abs(point - 0.5) < 0.05          # 显著为正
    zero = [rng.gauss(0, 1.0) for _ in range(300)]
    _, lo0, hi0 = paired_bootstrap_ci(zero, n_boot=2000, seed=43)
    assert lo0 < 0 < hi0                                # 跨零
    a = paired_bootstrap_ci(pos, n_boot=1000, seed=7)
    b = paired_bootstrap_ci(pos, n_boot=1000, seed=7)
    assert a == b                                       # 种子确定性
