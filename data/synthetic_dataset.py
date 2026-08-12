"""SyntheticVLMDataset — mock dataset matching CachedVLMDataset's schema.

Spec ref: linked_medusa_spec.md §6.2.

Hidden states are random Gaussian; tokens are uniform random over the
EFFECTIVE tokenizer vocab (151936 by default), not the padded lm_head dim
(152064). Real cache will only ever produce token ids < 151936 because the
extra rows correspond to padding slots the tokenizer never emits, so we
match that distribution.

Sequence lengths are sampled from `seq_len_range` so collate gets exercised
on padding. Lengths and per-sample seeds are pre-determined at __init__
time so repeat training runs see the same data.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticVLMDataset(Dataset):
    """Random fp16 hidden states + int64 tokens, deterministic per-sample.

    __getitem__(idx) returns:
        {
            "hidden": Tensor(L, hidden_dim) float16,
            "tokens": Tensor(L,)            int64,
        }
    """

    def __init__(
        self,
        num_samples: int = 200,
        seq_len_range: Tuple[int, int] = (50, 256),
        hidden_dim: int = 3584,
        vocab_size: int = 151936,
        seed: int = 42,
    ):
        if seq_len_range[0] < 4 or seq_len_range[1] < seq_len_range[0]:
            raise ValueError(f"bad seq_len_range: {seq_len_range}")
        self.num_samples = num_samples
        self.seq_len_range = tuple(seq_len_range)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.seed = seed

        # Pre-decide every sample's length deterministically; otherwise the
        # length distribution would shift between runs even if seed=42.
        g = torch.Generator().manual_seed(seed)
        lo, hi = seq_len_range
        self._lengths = torch.randint(lo, hi + 1, (num_samples,), generator=g).tolist()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        L = self._lengths[idx]
        # Per-sample seed offsets keep different samples independent and
        # also independent from the length-decision RNG above.
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
        hidden = torch.randn(L, self.hidden_dim, generator=g, dtype=torch.float32).to(
            torch.float16
        )
        tokens = torch.randint(
            0, self.vocab_size, (L,), generator=g, dtype=torch.int64
        )
        return {"hidden": hidden, "tokens": tokens}
