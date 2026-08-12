"""scripts/train.py — entrypoint for Linked Medusa training.

Spec ref: linked_medusa_spec.md §7 / §8.

Usage:
    # Full training on GPU
    python -m scripts.train --config config/linked_medusa_default.yaml

    # Local CPU smoke test (tiny model, a few steps, no AdamW8bit/AMP)
    python -m scripts.train --config config/linked_medusa_default.yaml --smoke

    # Override individual config keys (dotted-key=value, types auto-inferred)
    python -m scripts.train --config config/linked_medusa_default.yaml \
        --override training.batch_size=8 training.total_steps=20000

    # Resume
    python -m scripts.train --config config/linked_medusa_default.yaml \
        --override training.resume_from=/scratch/.../latest.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Project root on sys.path so absolute imports work whether called as
# `python -m scripts.train` or `python scripts/train.py`.
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from train.trainer import Trainer


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _coerce(value: str) -> Any:
    """Best-effort type coercion for CLI override RHS."""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    # list literal:  [1, 2, 3]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_coerce(x.strip()) for x in inner.split(",")]
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> None:
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"bad --override entry: {ov!r}  (expected key=value)")
        key, _, raw = ov.partition("=")
        path = key.strip().split(".")
        val = _coerce(raw.strip())
        node = cfg
        for k in path[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[path[-1]] = val
        print(f"[override] {key} = {val!r}")


def apply_smoke_overrides(cfg: Dict[str, Any]) -> None:
    """Apply CPU-friendly overrides so the training loop can run end-to-end
    on a login node in seconds.

    Smoke mode:
      - tiny dimensions (hidden=128, vocab=512)
      - disable lm_head init from disk (the real lm_head is 152064x3584 so it
        won't fit the tiny model)
      - tiny dataset
      - 10 steps, no eval, no save
      - optimizer=adamw (CPU-compatible)
      - use_amp=false (no GradScaler on CPU)
    """
    cfg.setdefault("model", {}).update(
        {
            "hidden_dim": 128,
            "vocab_size": 512,
            "num_blocks": 1,
            "expansion": 2,
            "init_lm_head_from_base": False,
        }
    )
    cfg.setdefault("dataset", {}).update(
        {
            "kind": "synthetic",
            "num_samples": 24,
            "seq_len_range": [50, 64],
            "max_length": 64,
            "val_fraction": 0.2,
        }
    )
    cfg.setdefault("training", {}).update(
        {
            "total_steps": 10,
            "batch_size": 4,
            "grad_accum_steps": 1,
            "num_workers": 0,
            "pin_memory": False,
            "optimizer": "adamw",     # CPU-friendly fallback (smoke only)
            "warmup_steps": 3,
            "use_amp": False,
        }
    )
    cfg.setdefault("logging", {}).update(
        {
            "log_every": 2,
            "eval_every": 5,
            "save_every": 0,        # no checkpoint files in smoke
            "output_dir": "/tmp/medusa_smoke",
        }
    )
    print("[smoke] applied CPU-friendly overrides (see scripts/train.py for the exact set)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config (e.g. config/linked_medusa_default.yaml)")
    p.add_argument("--smoke", action="store_true",
                   help="Apply CPU smoke-test overrides (tiny model, ~10 steps, no AdamW8bit).")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--override", nargs="*", default=[],
                   help="Dotted-key overrides, e.g. training.batch_size=8 training.total_steps=20000")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        apply_smoke_overrides(cfg)
    apply_overrides(cfg, args.override)

    trainer = Trainer(cfg, device=args.device)
    trainer.fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
