"""
Pretrained Swin Transformer Backbone & Classification Model.

This module provides:
1. Stem Adaptation (`adapt_input_stem`): Modifies the patch embedding projection stem (`patch_embed.proj`)
   to support variable input channel counts (3 for Kaggle single-modality images, 4 for BraTS early fusion).
2. Backbone Extractor (`SwinBackbone`): Feature extractor outputting raw embeddings [B, num_features].
3. Classification Head (`ClassificationHead`): LayerNorm -> Dropout -> Linear head.
4. Unified Classifier (`SwinClassifier`): End-to-end model combining Swin backbone and custom head to produce
   logits [B, num_classes].

Stem Adaptation Weight Rationale:
--------------------------------
Standard ImageNet pretrained vision backbones expect 3 RGB input channels. When expanding the input stem
to accept 4 MRI channels (T1, T1ce, T2, FLAIR):
1. The original pretrained weights for channels 0, 1, 2 (RGB) are copied directly into the new stem convolution
   weight tensor to preserve pretrained spatial feature detection capabilities.
2. The newly added 4th channel (index 3 for FLAIR) is initialized with the channel-wise mean of the 3 pretrained
   RGB weights (`old_weights.mean(dim=1, keepdim=True)`).
3. Rationale: Initializing the 4th channel with the RGB weight mean guarantees scale consistency with the
   pretrained filters and prevents gradient spikes or representation degradation during early training epochs.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import timm


def adapt_input_stem(model: nn.Module, input_channels: int = 4) -> nn.Module:
    """
    Modifies the patch embedding projection stem of a Swin Transformer model to accept a variable
    number of input channels while preserving pretrained RGB weights.

    Args:
        model (nn.Module): Swin Transformer model loaded from `timm`.
        input_channels (int): Target number of input channels (e.g., 4 for BraTS, 3 for Kaggle).

    Returns:
        nn.Module: Model with adapted stem patch embedding projection layer.
    """
    if not hasattr(model, "patch_embed") or not hasattr(model.patch_embed, "proj"):
        raise AttributeError("Model does not possess standard Swin Transformer 'patch_embed.proj' layer.")

    old_proj = model.patch_embed.proj
    old_in_chans = old_proj.in_channels

    if old_in_chans == input_channels:
        return model

    out_channels = old_proj.out_channels
    kernel_size = old_proj.kernel_size
    stride = old_proj.stride
    padding = old_proj.padding
    dilation = old_proj.dilation
    bias_flag = old_proj.bias is not None

    new_proj = nn.Conv2d(
        in_channels=input_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias_flag,
    )

    with torch.no_grad():
        old_weight = old_proj.weight  # Shape: [out_channels, old_in_chans, K_h, K_w]
        new_weight = new_proj.weight  # Shape: [out_channels, input_channels, K_h, K_w]

        min_chans = min(old_in_chans, input_channels)
        new_weight[:, :min_chans, :, :] = old_weight[:, :min_chans, :, :]

        if input_channels > old_in_chans:
            # RATIONALE: Initialize the 4th channel (FLAIR) with the average of pretrained RGB weights
            # to preserve scale matching and pretrained spatial frequency filters.
            mean_weight = old_weight.mean(dim=1, keepdim=True)
            for extra_idx in range(old_in_chans, input_channels):
                new_weight[:, extra_idx : extra_idx + 1, :, :] = mean_weight

        if bias_flag and old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)

    model.patch_embed.proj = new_proj
    if hasattr(model.patch_embed, "in_chans"):
        model.patch_embed.in_chans = input_channels

    return model


class SwinBackbone(nn.Module):
    """
    Swin Transformer Feature Extraction Backbone Module.

    Wraps a timm Swin Transformer backbone with modified input stem (supporting 3 or 4 channels)
    and removes the classification head to output raw feature representations.
    """

    def __init__(
        self,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        input_channels: int = 4,
        pretrained: bool = True,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.input_channels = input_channels
        self.pretrained = pretrained

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
        )

        self.initial_param_count = sum(p.numel() for p in self.backbone.parameters())

        if input_channels != 3:
            adapt_input_stem(self.backbone, input_channels=input_channels)

        self.num_parameters = sum(p.numel() for p in self.backbone.parameters())
        self.num_features = self.backbone.num_features  # 768 for swin_tiny

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor [B, C, H, W], got shape {x.shape}")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Input channel count {x.shape[1]} does not match model input_channels {self.input_channels}"
            )

        features = self.backbone(x)
        return features


class ClassificationHead(nn.Module):
    """
    Custom classification head for Swin Transformer models.

    Architecture:
        LayerNorm(in_features) -> Dropout(drop_rate) -> Linear(in_features, num_classes)
    """

    def __init__(self, in_features: int, num_classes: int, drop_rate: float = 0.2):
        """
        Args:
            in_features (int): Dimensionality of input feature vectors (e.g., 768).
            num_classes (int): Number of output classification targets.
            drop_rate (float): Dropout probability rate.
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(p=drop_rate)
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting feature vectors [B, in_features] to logits [B, num_classes].
        """
        x = self.norm(x)
        x = self.drop(x)
        return self.head(x)


class SwinClassifier(nn.Module):
    """
    Unified Swin Transformer Classification Model.

    Combines SwinBackbone (with stem adaptation for 3 or 4 channels) and a custom
    ClassificationHead to output logits of shape [batch_size, num_classes].
    """

    def __init__(
        self,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        input_channels: int = 4,
        num_classes: int = 2,
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ):
        """
        Args:
            backbone_name (str): Timm Swin Transformer architecture string.
            input_channels (int): Number of input channels (3 for Kaggle, 4 for BraTS).
            num_classes (int): Number of output classification targets (4 for Kaggle, 2 for BraTS).
            pretrained (bool): Load ImageNet pretrained weights for backbone.
            drop_rate (float): Dropout probability rate.
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.pretrained = pretrained

        # Feature extraction backbone (stem adapted)
        self.backbone = SwinBackbone(
            backbone_name=backbone_name,
            input_channels=input_channels,
            pretrained=pretrained,
            drop_rate=drop_rate,
        )

        self.num_features = self.backbone.num_features  # 768 for swin_tiny

        # Custom classification head
        self.head = ClassificationHead(
            in_features=self.num_features,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

        self.num_parameters = sum(p.numel() for p in self.parameters())

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass returning output logits [B, num_classes].

        Args:
            x (torch.Tensor): Input image batch of shape [B, input_channels, 224, 224].
            return_features (bool): If True, returns tuple (logits, features).

        Returns:
            torch.Tensor: Logits [B, num_classes] or Tuple (logits, features).
        """
        features = self.backbone(x)  # Shape: [B, 768]
        logits = self.head(features)  # Shape: [B, num_classes]

        if return_features:
            return logits, features

        return logits


def build_swin_backbone(
    backbone_name: str = "swin_tiny_patch4_window7_224",
    input_channels: int = 4,
    pretrained: bool = True,
    drop_rate: float = 0.0,
) -> SwinBackbone:
    return SwinBackbone(
        backbone_name=backbone_name,
        input_channels=input_channels,
        pretrained=pretrained,
        drop_rate=drop_rate,
    )


def build_swin_classifier(
    backbone_name: str = "swin_tiny_patch4_window7_224",
    input_channels: int = 4,
    num_classes: int = 2,
    pretrained: bool = True,
    drop_rate: float = 0.2,
) -> SwinClassifier:
    """
    Factory function to construct a SwinClassifier instance.
    """
    return SwinClassifier(
        backbone_name=backbone_name,
        input_channels=input_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
    )
