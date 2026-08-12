"""Linked Medusa head model package."""
from .resblock import MLPResBlock
from .linked_head import LinkedMedusaHead, LinkedMedusaHeads
from .init_utils import load_base_lm_head_weight

__all__ = [
    "MLPResBlock",
    "LinkedMedusaHead",
    "LinkedMedusaHeads",
    "load_base_lm_head_weight",
]
