"""
Multi-Modal Fusion Module for Brain Tumor Diagnosis.
"""

from src.fusion.fusion import (
    BraTSEarlyFusion,
    KagglePassThrough,
    fuse_brats_modalities,
    pass_through_kaggle,
    get_fusion_module,
)
from src.fusion.late_fusion import LateFusionModule

__all__ = [
    "BraTSEarlyFusion",
    "KagglePassThrough",
    "fuse_brats_modalities",
    "pass_through_kaggle",
    "get_fusion_module",
    "LateFusionModule",
]
