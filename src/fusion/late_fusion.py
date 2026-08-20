"""
Late Fusion Prototype Module for Multi-Modal MRI Brain Tumor Classification.

This module implements feature-level Late Fusion (LateFusionModule), where separate modality-specific
encoder streams independently extract feature representations from each of the 4 BraTS MRI modalities
(T1, T1ce, T2, FLAIR). The resulting feature vectors are fused via concatenation or element-wise averaging
prior to a classification head.

Comparison: Early Fusion vs. Late Fusion Architecture
------------------------------------------------------
1. Early Fusion (Day 11 - src/fusion/fusion.py):
   - Combines co-registered modalities at the input pixel level into a [4, H, W] tensor.
   - Shared Backbone: Uses a single Vision Transformer (Swin) backbone with a modified 4-channel input stem.
   - Parameter Count: ~O(N) backbone parameters. Highly parameter-efficient.
   - Cross-Modal Interaction: Joint spatial-spectral attention begins at layer 1.

2. Late Fusion (Day 12 - src/fusion/late_fusion.py):
   - Processes each modality through an independent feature encoder stream [B, 1, H, W] -> [B, D].
   - Fused Features: Merges 4 separate feature vectors via concatenation [B, 4*D] or averaging [B, D].
   - Parameter Count: ~O(4 * N) encoder parameters if using full backbones (4x higher parameter count).
   - Cross-Modal Interaction: Modalities remain isolated until late feature fusion.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class SingleModalityEncoder(nn.Module):
    """
    Lightweight CNN feature encoder for a single MRI modality [B, 1, H, W] -> [B, embed_dim].
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape [B, 1, H, W] or [B, H, W]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        feat = self.encoder(x)  # Shape: [B, embed_dim, 1, 1]
        return torch.flatten(feat, start_dim=1)  # Shape: [B, embed_dim]


class LateFusionModule(nn.Module):
    """
    Feature-level Late Fusion PyTorch module for BraTS 4-modality MRI scans.

    Expected Channel Order:
        Channel 0: T1
        Channel 1: T1ce
        Channel 2: T2
        Channel 3: FLAIR
    """

    MODALITIES = ("t1", "t1ce", "t2", "flair")

    def __init__(
        self,
        embed_dim: int = 128,
        num_classes: int = 2,
        fusion_mode: str = "concat",
        dropout_prob: float = 0.2,
    ):
        """
        Args:
            embed_dim (int): Feature vector dimension produced per modality encoder.
            num_classes (int): Number of target classification categories (default: 2 for LGG/HGG).
            fusion_mode (str): 'concat' (concatenates 4 features -> 4*embed_dim) or
                               'average' / 'mean' (averages 4 features -> embed_dim).
            dropout_prob (float): Dropout rate before classification head.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode.lower()
        if self.fusion_mode not in ("concat", "average", "mean"):
            raise ValueError(f"Unsupported fusion_mode '{fusion_mode}'. Choose 'concat' or 'average'.")

        # 4 independent encoder streams for the 4 BraTS modalities
        self.encoders = nn.ModuleDict({
            m: SingleModalityEncoder(in_channels=1, embed_dim=embed_dim) for m in self.MODALITIES
        })

        # Calculate input dimension for classification head
        if self.fusion_mode == "concat":
            in_classifier_dim = embed_dim * len(self.MODALITIES)
        else:
            in_classifier_dim = embed_dim

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_classifier_dim, num_classes)
        )

    def forward(
        self,
        modalities: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None,
        return_features: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for feature-level late fusion.

        Args:
            modalities: Can be:
                - Dict mapping modality names ('t1', 't1ce', 't2', 'flair') to [B, 1, H, W] or [B, H, W] tensors.
                - 4D Tensor of shape [B, 4, H, W] (stacked 4-channel tensor).
                - Keyword args (t1=..., t1ce=..., t2=..., flair=...).
            return_features (bool): If True, returns dict with logits, fused features, and per-modality features.

        Returns:
            torch.Tensor [B, num_classes] logits or Dict containing logits and intermediate features.
        """
        mod_dict: Dict[str, torch.Tensor] = {}

        if isinstance(modalities, torch.Tensor):
            if modalities.ndim == 4 and modalities.shape[1] == 4:
                # Split 4-channel tensor [B, 4, H, W] into per-modality tensors [B, 1, H, W]
                mod_dict = {
                    name: modalities[:, idx:idx+1, :, :]
                    for idx, name in enumerate(self.MODALITIES)
                }
            elif modalities.ndim == 4 and modalities.shape[0] == 4:
                # Unbatched 4-channel tensor [4, H, W] -> add batch dim [1, 4, H, W]
                modalities_batched = modalities.unsqueeze(0)
                mod_dict = {
                    name: modalities_batched[:, idx:idx+1, :, :]
                    for idx, name in enumerate(self.MODALITIES)
                }
            else:
                raise ValueError(f"Expected 4D tensor with 4 channels [B, 4, H, W], got shape {modalities.shape}")
        elif isinstance(modalities, dict):
            mod_dict = modalities
        elif kwargs:
            mod_dict = kwargs
        else:
            raise ValueError("No valid input provided to LateFusionModule.")

        # Extract features per modality
        per_modality_features: Dict[str, torch.Tensor] = {}
        feature_list = []
        for name in self.MODALITIES:
            if name not in mod_dict:
                raise KeyError(f"Missing required modality tensor for late fusion: '{name}'")
            x = mod_dict[name]
            feat = self.encoders[name](x)  # [B, embed_dim]
            per_modality_features[name] = feat
            feature_list.append(feat)

        # Merge features
        if self.fusion_mode == "concat":
            fused_features = torch.cat(feature_list, dim=1)  # [B, 4 * embed_dim]
        else:
            stacked = torch.stack(feature_list, dim=0)  # [4, B, embed_dim]
            fused_features = torch.mean(stacked, dim=0)  # [B, embed_dim]

        # Classification logits
        logits = self.classifier(fused_features)  # [B, num_classes]

        # NaN check
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("LateFusionModule output logits contain NaN or Inf values!")

        if return_features:
            return {
                "logits": logits,
                "fused_features": fused_features,
                "per_modality_features": per_modality_features,
            }

        return logits
