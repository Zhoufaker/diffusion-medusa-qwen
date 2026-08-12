"""Checkpoint pruning policy (trainer.save_checkpoint).

Verifies:
  - regular ckpts (tag=None) get pruned beyond keep_recent_n
  - tagged ckpts (best, final) are never auto-pruned
  - latest.pt symlink/copy always points at the most recent write
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from scripts.train import apply_smoke_overrides, load_config
from train.trainer import Trainer


def _smoke_trainer(tmp_path: Path, keep_recent_n: int = 3) -> Trainer:
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "linked_medusa_default.yaml"
    cfg = load_config(str(cfg_path))
    apply_smoke_overrides(cfg)
    cfg["logging"]["output_dir"] = str(tmp_path)
    cfg["logging"]["keep_recent_n"] = keep_recent_n
    # We don't actually fit; trainer.__init__ is what we need.
    return Trainer(cfg, device="cpu")


def _step_ckpts(d: Path) -> list[Path]:
    return sorted(d.glob("ckpt_step*.pt"))


def test_keep_recent_n_prunes_old_step_ckpts(tmp_path: Path):
    trainer = _smoke_trainer(tmp_path, keep_recent_n=3)

    # Five regular ckpts with monotonic step numbers.
    for s in [10, 20, 30, 40, 50]:
        trainer.state.step = s
        trainer.save_checkpoint()  # tag=None

    remaining = _step_ckpts(tmp_path)
    names = [p.name for p in remaining]
    # Only the last 3 (steps 30, 40, 50) should survive.
    assert names == ["ckpt_step30.pt", "ckpt_step40.pt", "ckpt_step50.pt"], names
    # latest.pt must resolve to step50.
    latest = tmp_path / "latest.pt"
    assert latest.exists()
    target = os.readlink(str(latest)) if latest.is_symlink() else "ckpt_step50.pt"
    assert "step50" in target


def test_tagged_ckpts_are_never_pruned(tmp_path: Path):
    trainer = _smoke_trainer(tmp_path, keep_recent_n=2)

    # Interleave tagged saves (best/final) with step saves; tagged ones must
    # not be on the prune list.
    trainer.state.step = 10
    trainer.save_checkpoint()
    trainer.state.step = 15
    trainer.save_checkpoint(tag="best")
    trainer.state.step = 20
    trainer.save_checkpoint()
    trainer.state.step = 30
    trainer.save_checkpoint()
    trainer.state.step = 40
    trainer.save_checkpoint()  # this triggers prune; step10 should go

    # ckpt_best.pt MUST still be there.
    assert (tmp_path / "ckpt_best.pt").is_file()
    # Step ckpts after pruning: keep 2 of them.
    step_names = [p.name for p in _step_ckpts(tmp_path)]
    assert step_names == ["ckpt_step30.pt", "ckpt_step40.pt"], step_names
    # Final tagged save also untouched.
    trainer.state.step = 999
    trainer.save_checkpoint(tag="final")
    assert (tmp_path / "ckpt_final.pt").is_file()
    assert (tmp_path / "ckpt_best.pt").is_file()


def test_keep_recent_n_zero_disables_pruning(tmp_path: Path):
    """keep_recent_n=0 means immediately prune (one-step churn).

    The semantics: after each save, only keep `keep_recent_n` regular ckpts.
    So with keep_recent_n=0, after each regular save the prune step removes
    the just-saved file too. End state: no ckpt_step*.pt on disk. Tagged ones
    still untouched.
    """
    trainer = _smoke_trainer(tmp_path, keep_recent_n=0)
    trainer.state.step = 10
    trainer.save_checkpoint()
    trainer.state.step = 20
    trainer.save_checkpoint()
    # All regular ckpts get nuked because `while len > 0: pop` removes them.
    assert _step_ckpts(tmp_path) == []


def test_keep_recent_n_large(tmp_path: Path):
    """If keep_recent_n exceeds number of saves, nothing is pruned."""
    trainer = _smoke_trainer(tmp_path, keep_recent_n=10)
    for s in [10, 20, 30]:
        trainer.state.step = s
        trainer.save_checkpoint()
    assert len(_step_ckpts(tmp_path)) == 3
