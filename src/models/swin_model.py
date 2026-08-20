"""
Pretrained Swin Transformer Backbone & Stem Adaptation Module.

This module loads ImageNet-pretrained Swin Transformer backbones via `timm` and adapts the initial
patch embedding stem layer (`patch_embed.proj`) to support variable input channel counts (e.g., 4-channel
BraTS multi-modal early fusion stacks [T1, T1ce, T2, FLAIR] or 3-channel single-modality Kaggle images).

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

    # If input channel count already matches, no stem modification required
    if old_in_chans == input_channels:
        return model

    # Extract original projection parameters
    out_channels = old_proj.out_channels
    kernel_size = old_proj.kernel_size
    stride = old_proj.stride
    padding = old_proj.padding
    dilation = old_proj.dilation
    bias_flag = old_proj.bias is not None

    # Create new Conv2d layer with adapted input channels
    new_proj = nn.Conv2d(
        in_channels=input_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias_flag,
    )

    # Initialize weights
    with torch.no_grad():
        old_weight = old_proj.weight  # Shape: [out_channels, old_in_chans, K_h, K_w]
        new_weight = new_proj.weight  # Shape: [out_channels, input_channels, K_h, K_w]

        # Copy over pretrained weights for shared channel indices
        min_chans = min(old_in_chans, input_channels)
        new_weight[:, :min_chans, :, :] = old_weight[:, :min_chans, :, :]

        # If expanding channels (e.g., from 3 RGB channels to 4 MRI modalities):
        if input_channels > old_in_chans:
            # RATIONALE: Initialize the 4th channel (FLAIR) with the average of pretrained RGB weights
            # to preserve scale matching and pretrained spatial frequency filters.
            mean_weight = old_weight.mean(dim=1, keepdim=True)  # Shape: [out_channels, 1, K_h, K_w]
            for extra_idx in range(old_in_chans, input_channels):
                new_weight[:, extra_idx : extra_idx + 1, :, :] = mean_weight

        # Copy bias if present
        if bias_flag and old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)

    # Replace stem layer in Swin Transformer model
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
        """
        Args:
            backbone_name (str): Backbone architecture name in `timm`.
            input_channels (int): Number of input channels (e.g., 4 for BraTS early fusion, 3 for Kaggle).
            pretrained (bool): Whether to load ImageNet pretrained weights.
            drop_rate (float): Dropout probability rate.
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.input_channels = input_channels
        self.pretrained = pretrained

        # Load backbone with num_classes=0 to detach classification head
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
        )

        # Record initial parameter count before stem modification
        self.initial_param_count = sum(p.numel() for p in self.backbone.parameters())

        # Adapt input stem if channel count differs from standard 3-channel RGB
        if input_channels != 3:
            adapt_input_stem(self.backbone, input_channels=input_channels)

        # Record final parameter count after stem adaptation
        self.num_parameters = sum(p.numel() for p in self.backbone.parameters())
        self.num_features = self.backbone.num_features  # 768 for swin_tiny

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing feature embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W] where C is `input_channels`.

        Returns:
            torch.Tensor: Feature embeddings of shape [B, num_features] (e.g., [B, 768]).
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor [B, C, H, W], got shape {x.shape}")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Input channel count {x.shape[1]} does not match model input_channels {self.input_channels}"
            )

        features = self.backbone(x)
        return features


def build_swin_backbone(
    backbone_name: str = "swin_tiny_patch4_window7_224",
    input_channels: int = 4,
    pretrained: bool = True,
    drop_rate: float = 0.0,
) -> SwinBackbone:
    """
    Factory function to construct a SwinBackbone instance.

    Args:
        backbone_name (str): Timm Swin Transformer architecture string.
        input_channels (int): Input channel count (3 or 4).
        pretrained (bool): Load ImageNet pretrained weights.
        drop_rate (float): Dropout probability.

    Returns:
        SwinBackbone: Instantiated feature extraction model.
    """
    return SwinBackbone(
        backbone_name=backbone_name,
        input_channels=input_channels,
        pretrained=pretrained,
        drop_rate=drop_rate,
    )
