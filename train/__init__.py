"""Training-related primitives (loss, scheduler helpers, eval metrics)."""
from .loss import compute_loss, IGNORE_INDEX
from .scheduler import cosine_warmup_schedule
from .evaluate import evaluate
from .trainer import Trainer, build_dataset, build_optimizer

__all__ = [
    "compute_loss",
    "IGNORE_INDEX",
    "cosine_warmup_schedule",
    "evaluate",
    "Trainer",
    "build_dataset",
    "build_optimizer",
]
