"""
Transfer learning module for DeFungi dataset classification.
"""

import sys
from pathlib import Path

# Ensure current package directory is in sys.path
_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

try:
    from .dataset import DeFungiDataset, get_dataloaders, get_transforms, find_dataset_dir
    from .model import build_resnet18, get_parameter_counts, unfreeze_backbone
except (ImportError, ValueError):
    from dataset import DeFungiDataset, get_dataloaders, get_transforms, find_dataset_dir
    from model import build_resnet18, get_parameter_counts, unfreeze_backbone

__all__ = [
    "DeFungiDataset",
    "get_dataloaders",
    "get_transforms",
    "find_dataset_dir",
    "build_resnet18",
    "get_parameter_counts",
    "unfreeze_backbone",
]
