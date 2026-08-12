"""build_dataset + _parse_eval_split tests (spec §6, trainer.py).

Guards the (train, val) factory entry point used by Trainer.__init__. The
specific behaviour we care about for production:

  - dataset.eval_split = "last_2000" must yield a deterministic, numerically-
    last 2000 samples as val, and the rest as train. NO RNG seeding required.
  - dataset.eval_split = null falls back to val_set=None (Trainer applies
    val_fraction random_split itself).
  - Synthetic with eval_split=last_N partitions by index (uses Subset).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from train.trainer import _parse_eval_split, build_dataset


def _make_fake_cache(tmp: Path, n: int, hidden_dim: int = 4, L: int = 8) -> None:
    for stem in range(n):
        torch.save(
            {
                "hidden": torch.zeros(L, hidden_dim, dtype=torch.float16) + float(stem),
                "tokens": torch.arange(L, dtype=torch.int64),
            },
            str(tmp / f"{stem}.pt"),
        )


# ---------------------------------------------------------------------------
# _parse_eval_split
# ---------------------------------------------------------------------------


def test_parse_eval_split_none():
    assert _parse_eval_split(None) is None
    assert _parse_eval_split(0) is None
    assert _parse_eval_split("none") is None


def test_parse_eval_split_int_passthrough():
    assert _parse_eval_split(2000) == 2000
    assert _parse_eval_split(1) == 1


def test_parse_eval_split_last_n_string():
    assert _parse_eval_split("last_2000") == 2000
    assert _parse_eval_split("last_5") == 5


def test_parse_eval_split_bad_format_raises():
    with pytest.raises(ValueError):
        _parse_eval_split("garbage")
    with pytest.raises(ValueError):
        _parse_eval_split("last_garbage")
    with pytest.raises(ValueError):
        _parse_eval_split([1, 2, 3])


# ---------------------------------------------------------------------------
# build_dataset — cached
# ---------------------------------------------------------------------------


def _base_cfg(cache_dir: str, eval_split=None) -> dict:
    return {
        "seed": 42,
        "model": {"hidden_dim": 4, "vocab_size": 32},
        "dataset": {
            "kind": "cached",
            "cache_dir": cache_dir,
            "max_length": 64,
            "eval_split": eval_split,
            "val_fraction": 0.0,
        },
    }


def test_build_dataset_cached_with_last_n(tmp_path: Path):
    _make_fake_cache(tmp_path, n=10)
    cfg = _base_cfg(str(tmp_path), eval_split="last_3")
    train, val = build_dataset(cfg)
    assert len(train) == 7
    assert len(val) == 3
    # The last 3 are stems 7, 8, 9; first 7 are stems 0..6.
    assert [train.stem(i) for i in range(7)] == [0, 1, 2, 3, 4, 5, 6]
    assert [val.stem(i) for i in range(3)] == [7, 8, 9]


def test_build_dataset_cached_int_eval_split(tmp_path: Path):
    _make_fake_cache(tmp_path, n=10)
    cfg = _base_cfg(str(tmp_path), eval_split=4)
    train, val = build_dataset(cfg)
    assert len(train) == 6
    assert len(val) == 4
    assert [val.stem(i) for i in range(4)] == [6, 7, 8, 9]


def test_build_dataset_cached_no_eval_split_returns_none_val(tmp_path: Path):
    """eval_split=None -> val is None, caller does random_split."""
    _make_fake_cache(tmp_path, n=5)
    cfg = _base_cfg(str(tmp_path), eval_split=None)
    train, val = build_dataset(cfg)
    assert len(train) == 5
    assert val is None


# ---------------------------------------------------------------------------
# build_dataset — synthetic
# ---------------------------------------------------------------------------


def _synth_cfg(eval_split=None) -> dict:
    return {
        "seed": 42,
        "model": {"hidden_dim": 16, "vocab_size": 64},
        "dataset": {
            "kind": "synthetic",
            "num_samples": 20,
            "seq_len_range": [8, 12],
            "eval_split": eval_split,
            "val_fraction": 0.0,
        },
    }


def test_build_dataset_synthetic_with_last_n():
    cfg = _synth_cfg(eval_split="last_4")
    train, val = build_dataset(cfg)
    assert len(train) == 16
    assert len(val) == 4


def test_build_dataset_synthetic_no_split_returns_none_val():
    cfg = _synth_cfg(eval_split=None)
    train, val = build_dataset(cfg)
    assert len(train) == 20
    assert val is None
