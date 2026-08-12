"""Dataset and collate package for Linked Medusa training."""
from .synthetic_dataset import SyntheticVLMDataset
from .cached_dataset import CachedVLMDataset
from .collate import collate_fn

__all__ = ["SyntheticVLMDataset", "CachedVLMDataset", "collate_fn"]
