"""CachedVLMDataset — load precomputed VLM hidden+token cache from disk.

Spec ref: linked_medusa_spec.md §6.1 + §6.2.

Cache layout (one .pt file per sample):

    /scratch/li96/mz9869/cached_data/qwen25vl_long/
    ├── manifest.json            (optional — skipped)
    ├── 0.pt
    ├── 1.pt
    ├── ...
    └── 46613.pt

Each .pt is a torch.save dict:

    {
        'hidden': Tensor(L, 3584)  fp16,    # base model's last-layer hidden states
        'tokens': Tensor(L,)       int64,   # input tokens at the matching positions
    }

L varies per sample (~65 .. >256). Files longer than `max_length` are
truncated to the first `max_length` rows.

PITFALL — file ordering
=======================
`os.listdir`, `glob.glob`, and `pathlib.Path.iterdir()` all return results in
filesystem order, which is effectively unsorted. Naïve sort gives LEXICAL
order:

    sorted(['0.pt', '10.pt', '2.pt']) == ['0.pt', '10.pt', '2.pt']

That would make `dataset[0]` map to the file `0.pt` but `dataset[1]` map to
`10.pt` — silently shifting indices and breaking any logging/debugging that
references "sample 5". We sort by `int(stem)` instead so indices match file
numbers (`dataset[k] <-> k.pt`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CachedVLMDataset(Dataset):
    """Disk-backed dataset reading one .pt per __getitem__ call.

    No in-memory caching — at 46k * ~2 MiB the full cache is ~92 GB and won't
    fit on a V100 node's RAM. Rely on the OS page cache + DataLoader workers.
    """

    def __init__(
        self,
        cache_dir: str,
        max_length: int = 256,
        hidden_dim_expected: int = 3584,
        eval_split_size: int = 0,
        split: str = "all",
    ):
        """
        Args:
            cache_dir, max_length, hidden_dim_expected: as before.
            eval_split_size: number of last-N files (by numeric stem order) to
                reserve as the eval set. If 0, no split.
            split: which portion to expose. One of:
                'all'   - all files (default, ignores eval_split_size)
                'train' - first len-eval_split_size files
                'eval'  - last eval_split_size files
            The numeric-sort guarantee means eval and train are reproducible
            across runs without an RNG seed, and a given <stem>.pt always lands
            in the same split given the same eval_split_size.
        """
        if split not in {"all", "train", "eval"}:
            raise ValueError(f"split must be 'all'|'train'|'eval'; got {split!r}")
        if eval_split_size < 0:
            raise ValueError(f"eval_split_size must be >= 0; got {eval_split_size}")
        if split != "all" and eval_split_size == 0:
            # Asking for train/eval slice but no split point defined - treat as
            # all-train (the eval set is empty).
            pass

        self.cache_dir = Path(cache_dir)
        self.max_length = int(max_length)
        self.hidden_dim_expected = int(hidden_dim_expected)
        self.eval_split_size = int(eval_split_size)
        self.split = split

        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"cache_dir does not exist: {self.cache_dir}")

        # Numeric sort by stem (see module docstring for why).
        files: List[Path] = []
        for p in self.cache_dir.iterdir():
            if p.suffix != ".pt":
                continue
            try:
                int(p.stem)
            except ValueError:
                # Skip non-numeric .pt files (none expected, but defensive).
                continue
            files.append(p)

        if not files:
            raise FileNotFoundError(
                f"no <int>.pt files found under {self.cache_dir}. "
                f"Did the rsync finish? Did you point at the right qwen25vl_long/?"
            )

        files.sort(key=lambda p: int(p.stem))

        # Apply split.
        if self.split == "train" and self.eval_split_size > 0:
            if self.eval_split_size >= len(files):
                raise ValueError(
                    f"eval_split_size={self.eval_split_size} >= total files "
                    f"{len(files)}; nothing left for train."
                )
            files = files[: -self.eval_split_size]
        elif self.split == "eval" and self.eval_split_size > 0:
            if self.eval_split_size > len(files):
                raise ValueError(
                    f"eval_split_size={self.eval_split_size} > total files {len(files)}"
                )
            files = files[-self.eval_split_size :]
        # split == 'all' or eval_split_size == 0 -> no slice

        self._files: List[Path] = files
        self._stems: List[int] = [int(p.stem) for p in files]

    # --- introspection -------------------------------------------------------

    def __len__(self) -> int:
        return len(self._files)

    def stem(self, idx: int) -> int:
        """Return the original numeric filename stem for index idx."""
        return self._stems[idx]

    def path(self, idx: int) -> Path:
        return self._files[idx]

    # --- core ----------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        p = self._files[idx]
        # weights_only=True is safe: each file is just a dict of tensors.
        d = torch.load(p, map_location="cpu", weights_only=True)

        if not isinstance(d, dict) or "hidden" not in d or "tokens" not in d:
            raise RuntimeError(
                f"{p}: expected dict with 'hidden' and 'tokens'; got "
                f"{type(d).__name__} with keys "
                f"{list(d.keys()) if isinstance(d, dict) else 'n/a'}"
            )

        hidden, tokens = d["hidden"], d["tokens"]

        if hidden.ndim != 2 or tokens.ndim != 1:
            raise RuntimeError(
                f"{p}: bad shapes hidden={tuple(hidden.shape)} tokens={tuple(tokens.shape)}"
            )
        if hidden.shape[0] != tokens.shape[0]:
            raise RuntimeError(
                f"{p}: hidden L={hidden.shape[0]} != tokens L={tokens.shape[0]}"
            )
        if hidden.shape[1] != self.hidden_dim_expected:
            raise RuntimeError(
                f"{p}: hidden_dim {hidden.shape[1]} != expected {self.hidden_dim_expected}"
            )

        # Normalize dtype (defensive — cache should already be fp16/int64).
        if hidden.dtype != torch.float16:
            hidden = hidden.to(torch.float16)
        if tokens.dtype != torch.int64:
            tokens = tokens.to(torch.int64)

        if hidden.shape[0] > self.max_length:
            hidden = hidden[: self.max_length].contiguous()
            tokens = tokens[: self.max_length].contiguous()

        return {"hidden": hidden, "tokens": tokens}
