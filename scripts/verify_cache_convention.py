"""verify_cache_convention.py — Phase B prerequisite check.

Spec ref: linked_medusa_spec.md §11.1.

The loss code in `train/loss.py` assumes INPUT-token convention:

    tokens[t] is the INPUT token fed to the base model at step t.
    hidden[t] is the base model's last-layer hidden AFTER processing tokens[t].
    --> prediction target for hidden[t] is tokens[t+1].

The cache layout is fixed by whoever generated it. Before training, we
verify that the convention matches by feeding hidden[t] through the base
lm_head and checking what it argmax-predicts.

Decision rule:
    if argmax(hidden[t] @ W^T) == tokens[t+1]   -> INPUT convention (loss is correct)
    if argmax(hidden[t] @ W^T) == tokens[t]     -> TARGET convention (need to fix offsets in loss.py)
    else                                        -> base lm_head disagrees with cache (something is wrong)

We sample a few positions per sample to make a stable call. The base model
is large (151k vocab) so on properly-aligned cache we expect >=90% of
"clean" positions (skipping the very first one where there's no t-1
context) to argmax to tokens[t+1].
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Allow `python scripts/verify_cache_convention.py` and `python -m scripts.verify_cache_convention`.
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from model.init_utils import load_base_lm_head_weight

DEFAULT_TEST_DIR = "/scratch/li96/mz9869/cached_data_test/qwen25vl_long"


def _list_pt_files_numeric(d: Path) -> List[Path]:
    files = []
    for p in d.iterdir():
        if p.suffix != ".pt":
            continue
        try:
            int(p.stem)
        except ValueError:
            continue
        files.append(p)
    files.sort(key=lambda p: int(p.stem))
    return files


@torch.no_grad()
def verify_one_sample(
    path: Path,
    W: torch.Tensor,
    positions_to_check: int = 16,
) -> Dict[str, object]:
    """Run the §11.1 check on a single cache file.

    Returns a dict with the raw counts so the caller can aggregate."""
    d = torch.load(path, map_location="cpu", weights_only=True)
    hidden = d["hidden"].float()   # (L, H)  — cast for stable matmul
    tokens = d["tokens"]           # (L,)    int64
    L = hidden.shape[0]
    V, H = W.shape
    if hidden.shape[1] != H:
        raise RuntimeError(f"{path}: hidden_dim mismatch {hidden.shape[1]} vs lm_head H={H}")
    if L < 2:
        raise RuntimeError(f"{path}: L={L} too short to test")

    # We use positions 0..L-2 (every t with a t+1 to compare against).
    # Sample evenly to keep the cost reasonable for long sequences.
    n_check = min(positions_to_check, L - 1)
    idx = torch.linspace(0, L - 2, steps=n_check, dtype=torch.int64).tolist()

    hits_next, hits_self, mismatches = 0, 0, 0
    detail_rows: List[Tuple[int, int, int, int, int]] = []  # (t, pred, tokens[t], tokens[t+1], topk hit)

    for t in idx:
        logits = hidden[t] @ W.t()   # (V,)
        pred = int(logits.argmax().item())
        tok_t = int(tokens[t].item())
        tok_t1 = int(tokens[t + 1].item())
        if pred == tok_t1:
            hits_next += 1
            hits_kind = "next"
        elif pred == tok_t:
            hits_self += 1
            hits_kind = "self"
        else:
            mismatches += 1
            hits_kind = "other"
        detail_rows.append((t, pred, tok_t, tok_t1, hits_kind))

    return {
        "path": str(path),
        "L": L,
        "n_checked": n_check,
        "hits_next": hits_next,
        "hits_self": hits_self,
        "mismatches": mismatches,
        "details": detail_rows,
    }


def render_sample_report(r: Dict[str, object]) -> str:
    head = (
        f"\n=== {Path(r['path']).name}  L={r['L']}, checked {r['n_checked']} positions ===\n"
        f"  argmax == tokens[t+1] (INPUT conv.)  : {r['hits_next']:>3d} / {r['n_checked']}\n"
        f"  argmax == tokens[t]   (TARGET conv.) : {r['hits_self']:>3d} / {r['n_checked']}\n"
        f"  neither                              : {r['mismatches']:>3d} / {r['n_checked']}\n"
    )
    table = "  t     pred       tokens[t]   tokens[t+1]   match\n"
    for t, pred, tok_t, tok_t1, kind in r["details"][:8]:
        table += f"  {t:>4d}  {pred:>8d}  {tok_t:>10d}   {tok_t1:>10d}   {kind}\n"
    if len(r["details"]) > 8:
        table += f"  ... ({len(r['details']) - 8} more rows)\n"
    return head + table


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache-dir",
        default=DEFAULT_TEST_DIR,
        help="Directory containing <int>.pt files. Default: cached_data_test (3 samples).",
    )
    p.add_argument(
        "--lm-head",
        default="/scratch/li96/mz9869/medusa_assets/base_lm_head.safetensors",
        help="Path to base_lm_head.safetensors (from scripts/extract_base_lm_head.py).",
    )
    p.add_argument(
        "--positions-per-sample",
        type=int,
        default=16,
        help="How many positions per sample to check (evenly spaced).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on number of samples to check. Default: all.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Pass threshold for INPUT-convention ratio over all checked positions (default 0.80).",
    )
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    print(f"[verify §11.1] cache_dir = {cache_dir}")
    print(f"[verify §11.1] lm_head   = {args.lm_head}")
    print(f"[verify §11.1] positions/sample = {args.positions_per_sample}")
    print(f"[verify §11.1] pass threshold (input-conv fraction) = {args.threshold}\n")

    files = _list_pt_files_numeric(cache_dir)
    if args.max_samples is not None:
        files = files[: args.max_samples]
    if not files:
        print(f"[verify §11.1] FAIL: no <int>.pt files found under {cache_dir}")
        return 2
    print(f"[verify §11.1] found {len(files)} sample(s): {[p.name for p in files]}")

    W = load_base_lm_head_weight(args.lm_head)
    print(f"[verify §11.1] lm_head shape = {tuple(W.shape)} dtype = {W.dtype}\n")
    # cast to fp32 once for stable per-sample matmul; weight is ~1 GB in fp16
    W = W.float()

    total = {"hits_next": 0, "hits_self": 0, "mismatches": 0, "n_checked": 0}
    per_sample = []
    for f in files:
        r = verify_one_sample(f, W, positions_to_check=args.positions_per_sample)
        per_sample.append(r)
        print(render_sample_report(r))
        for k in ("hits_next", "hits_self", "mismatches", "n_checked"):
            total[k] += r[k]

    n = total["n_checked"]
    frac_next = total["hits_next"] / max(1, n)
    frac_self = total["hits_self"] / max(1, n)
    frac_other = total["mismatches"] / max(1, n)
    print("=" * 64)
    print(f"[verify §11.1] aggregate over {len(files)} samples, {n} positions:")
    print(f"               INPUT-convention  (argmax==tokens[t+1]): {total['hits_next']:>4d}  ({frac_next*100:5.1f}%)")
    print(f"               TARGET-convention (argmax==tokens[t]  ): {total['hits_self']:>4d}  ({frac_self*100:5.1f}%)")
    print(f"               neither                                : {total['mismatches']:>4d}  ({frac_other*100:5.1f}%)")
    print("=" * 64)

    if frac_next >= args.threshold:
        print(f"\nVERDICT: INPUT convention CONFIRMED (>= {args.threshold*100:.0f}%). "
              "train/loss.py offsets are correct, NO CHANGES needed.")
        return 0
    if frac_self >= args.threshold:
        print(f"\nVERDICT: TARGET convention detected. You MUST update train/loss.py: "
              "shift pred/target offsets by `k` instead of `k+1` (see spec §11.1).")
        return 1
    print(f"\nVERDICT: AMBIGUOUS. Input-conv {frac_next*100:.1f}%, target-conv "
          f"{frac_self*100:.1f}%, neither {frac_other*100:.1f}%. "
          "Inspect the per-sample table — possibly a base_lm_head / cache mismatch.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
