"""CachedVLMDataset tests (spec §6.1, §6.2).

Two flavours:
  - synthetic tmp-dir tests (always run): exercise numeric sort, truncation,
    bad-file handling on a small in-memory cache we build in tmp.
  - real test-cache tests (explicit external-data gate): point at
    /scratch/li96/mz9869/cached_data_test/qwen25vl_long/{0,16761,37899}.pt
    and verify the dataset can decode the actual supervisor-provided format.
    That directory is a training-phase teacher-hidden cache on /scratch and is
    subject to scratch lifecycle purges; it is not part of the release evidence
    chain (O0 / decoding), so its absence gates these two cases only.

The numeric-sort tests are the critical ones; the rest are guard rails.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from data import CachedVLMDataset

REAL_TEST_DIR = Path("/scratch/li96/mz9869/cached_data_test/qwen25vl_long")
REAL_AVAILABLE = REAL_TEST_DIR.is_dir() and any(REAL_TEST_DIR.glob("*.pt"))

# Explicit gate in the same style as tests/test_max_new_cap_gpu.py: the reason
# names the missing data category, so a skip line in a release log states what
# is absent and cannot be read as a silently disabled negative control.
REAL_DATA_REASON = (
    "external training-phase data required: teacher-hidden cache "
    f"{REAL_TEST_DIR}/*.pt (scratch lifecycle; not in O0/decoding evidence chain)"
)
requires_real_train_cache = pytest.mark.skipif(not REAL_AVAILABLE, reason=REAL_DATA_REASON)


def _make_fake_cache(tmp: Path, lengths: dict[int, int], hidden_dim: int = 3584) -> None:
    """Create <stem>.pt for each (stem, L) in lengths under tmp/."""
    for stem, L in lengths.items():
        hidden = torch.zeros(L, hidden_dim, dtype=torch.float16)
        # Encode the stem and position into the tensor so we can identify which
        # file a sample originated from.
        hidden[:, 0] = float(stem)
        tokens = torch.arange(L, dtype=torch.int64) + stem * 1000
        torch.save({"hidden": hidden, "tokens": tokens}, str(tmp / f"{stem}.pt"))


# ---------------------------------------------------------------------------
# Synthetic tmp-dir tests
# ---------------------------------------------------------------------------


def test_numeric_sort_not_lexicographic(tmp_path: Path):
    """Critical: dataset[k] must map to <k>.pt, not the lexicographically k-th file."""
    stems = {0: 8, 1: 8, 2: 8, 9: 8, 10: 8, 100: 8, 1000: 8}
    _make_fake_cache(tmp_path, stems, hidden_dim=4)
    ds = CachedVLMDataset(str(tmp_path), max_length=64, hidden_dim_expected=4)
    expected_order = [0, 1, 2, 9, 10, 100, 1000]
    assert [ds.stem(i) for i in range(len(ds))] == expected_order
    # Index k -> file <expected_order[k]>.pt: verify by decoding the stem we
    # encoded into hidden[:, 0].
    for k, want_stem in enumerate(expected_order):
        item = ds[k]
        assert int(item["hidden"][0, 0].item()) == want_stem, (
            f"ds[{k}] returned stem {int(item['hidden'][0, 0].item())}, expected {want_stem}"
        )


def test_skips_manifest_and_non_int_files(tmp_path: Path):
    _make_fake_cache(tmp_path, {0: 4, 1: 4, 2: 4}, hidden_dim=4)
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "junk.pt").write_text("not a torch file")
    ds = CachedVLMDataset(str(tmp_path), max_length=64, hidden_dim_expected=4)
    assert len(ds) == 3
    assert [ds.stem(i) for i in range(3)] == [0, 1, 2]


def test_truncates_to_max_length(tmp_path: Path):
    _make_fake_cache(tmp_path, {0: 300}, hidden_dim=4)
    ds = CachedVLMDataset(str(tmp_path), max_length=128, hidden_dim_expected=4)
    item = ds[0]
    assert item["hidden"].shape == (128, 4)
    assert item["tokens"].shape == (128,)
    # tokens were arange(L) + 0; truncation keeps the first 128
    assert item["tokens"][0].item() == 0
    assert item["tokens"][-1].item() == 127


def test_short_sample_passes_through(tmp_path: Path):
    _make_fake_cache(tmp_path, {7: 65}, hidden_dim=4)
    ds = CachedVLMDataset(str(tmp_path), max_length=256, hidden_dim_expected=4)
    item = ds[0]
    assert item["hidden"].shape == (65, 4)
    assert item["tokens"].shape == (65,)


def test_empty_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CachedVLMDataset(str(tmp_path), max_length=64)


def test_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        CachedVLMDataset("/this/path/should/not/exist/please", max_length=64)


def test_bad_file_raises_at_getitem(tmp_path: Path):
    """A .pt file with wrong keys should raise a clear error at __getitem__,
    not silently return garbage."""
    L = 8
    torch.save(
        {"hidden": torch.zeros(L, 4, dtype=torch.float16)},  # missing 'tokens'
        str(tmp_path / "0.pt"),
    )
    ds = CachedVLMDataset(str(tmp_path), max_length=64, hidden_dim_expected=4)
    with pytest.raises(RuntimeError, match="tokens"):
        _ = ds[0]


def test_dtype_normalization(tmp_path: Path):
    """If a cache file's hidden is fp32, the dataset must cast it to fp16."""
    L = 4
    torch.save(
        {
            "hidden": torch.zeros(L, 4, dtype=torch.float32),
            "tokens": torch.arange(L, dtype=torch.int64),
        },
        str(tmp_path / "0.pt"),
    )
    ds = CachedVLMDataset(str(tmp_path), max_length=64, hidden_dim_expected=4)
    item = ds[0]
    assert item["hidden"].dtype == torch.float16


# ---------------------------------------------------------------------------
# eval_split / split tests
# ---------------------------------------------------------------------------


def test_eval_split_last_n_partitions_correctly(tmp_path: Path):
    """eval_split_size=N + split='train'/'eval' partitions by numeric stem.

    Files 0..9 with eval_split_size=3:
      train -> [0, 1, 2, 3, 4, 5, 6]  (7 files)
      eval  -> [7, 8, 9]              (3 files)
    """
    stems = {i: 8 for i in range(10)}
    _make_fake_cache(tmp_path, stems, hidden_dim=4)

    train = CachedVLMDataset(
        str(tmp_path), max_length=64, hidden_dim_expected=4,
        eval_split_size=3, split="train",
    )
    evalset = CachedVLMDataset(
        str(tmp_path), max_length=64, hidden_dim_expected=4,
        eval_split_size=3, split="eval",
    )
    assert len(train) == 7
    assert len(evalset) == 3
    assert [train.stem(i) for i in range(7)] == [0, 1, 2, 3, 4, 5, 6]
    assert [evalset.stem(i) for i in range(3)] == [7, 8, 9]
    # No overlap; together they cover all files.
    overlap = set(train.stem(i) for i in range(7)) & set(evalset.stem(i) for i in range(3))
    assert overlap == set()


def test_eval_split_uses_numeric_not_lex_order(tmp_path: Path):
    """The split point must respect numeric, not lexicographic, ordering.

    Lex: ['0', '1', '10', '11', '2', '3', '4', '5', '6', '7', '8', '9']
    Numeric: [0..11]
    With eval_split_size=2, eval should contain 10 and 11 (numerically last),
    not '8' and '9' (lex last).
    """
    stems = {i: 8 for i in range(12)}
    _make_fake_cache(tmp_path, stems, hidden_dim=4)
    evalset = CachedVLMDataset(
        str(tmp_path), max_length=64, hidden_dim_expected=4,
        eval_split_size=2, split="eval",
    )
    assert [evalset.stem(i) for i in range(2)] == [10, 11]


def test_eval_split_size_zero_with_split_returns_all(tmp_path: Path):
    """split='train' with eval_split_size=0 returns the full file list."""
    stems = {i: 8 for i in range(5)}
    _make_fake_cache(tmp_path, stems, hidden_dim=4)
    ds = CachedVLMDataset(
        str(tmp_path), max_length=64, hidden_dim_expected=4,
        eval_split_size=0, split="train",
    )
    assert len(ds) == 5


def test_eval_split_too_large_raises(tmp_path: Path):
    stems = {i: 8 for i in range(5)}
    _make_fake_cache(tmp_path, stems, hidden_dim=4)
    with pytest.raises(ValueError, match=r"eval_split_size"):
        CachedVLMDataset(
            str(tmp_path), max_length=64, hidden_dim_expected=4,
            eval_split_size=5, split="train",
        )
    with pytest.raises(ValueError, match=r"eval_split_size"):
        CachedVLMDataset(
            str(tmp_path), max_length=64, hidden_dim_expected=4,
            eval_split_size=10, split="eval",
        )


def test_eval_split_bad_split_value_raises(tmp_path: Path):
    stems = {i: 8 for i in range(5)}
    _make_fake_cache(tmp_path, stems, hidden_dim=4)
    with pytest.raises(ValueError, match=r"split"):
        CachedVLMDataset(
            str(tmp_path), max_length=64, hidden_dim_expected=4,
            eval_split_size=2, split="invalid",
        )


# ---------------------------------------------------------------------------
# Real test-cache tests (explicit gate: external training-phase data required)
# ---------------------------------------------------------------------------


@requires_real_train_cache
def test_real_test_cache_loads():
    ds = CachedVLMDataset(str(REAL_TEST_DIR), max_length=256, hidden_dim_expected=3584)
    assert len(ds) == 3
    assert [ds.stem(i) for i in range(3)] == [0, 16761, 37899]
    # First sample is 256 long; second/third are short (verified in inspection).
    item0 = ds[0]
    assert item0["hidden"].shape == (256, 3584)
    assert item0["hidden"].dtype == torch.float16
    assert item0["tokens"].dtype == torch.int64
    item1 = ds[1]
    assert item1["hidden"].shape == (65, 3584)
    item2 = ds[2]
    assert item2["hidden"].shape == (68, 3584)


@requires_real_train_cache
def test_real_test_cache_collate():
    """End-to-end: dataset -> collate_fn -> batched tensors."""
    from data import collate_fn

    ds = CachedVLMDataset(str(REAL_TEST_DIR), max_length=256, hidden_dim_expected=3584)
    batch = collate_fn([ds[0], ds[1], ds[2]])
    assert batch["hidden"].shape == (3, 256, 3584)
    assert batch["tokens"].shape == (3, 256)
    assert batch["attention_mask"].shape == (3, 256)
    # The short samples (L=65, 68) should be padded; their last positions
    # must be ignore_index=-100.
    assert batch["tokens"][1, 65:].eq(-100).all()
    assert batch["tokens"][2, 68:].eq(-100).all()
    # And the first sample (L=256) should be fully valid.
    assert batch["attention_mask"][0].all()
