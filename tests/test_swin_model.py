"""
Test and verification script for Pretrained Swin Transformer Backbone & Input Stem Adaptation.

This script:
1. Verifies parameter counts before and after stem adaptation from 3 to 4 channels.
2. Tests standard 3-channel Swin backbone with a dummy input [1, 3, 224, 224] (Kaggle dataset setting).
3. Tests adapted 4-channel Swin backbone with a dummy input [1, 4, 224, 224] (BraTS early fusion setting).
4. Verifies output feature embeddings shape [1, 768], data type (torch.float32), and zero NaN/Inf errors.
5. Verifies 4th channel weight initialization rationale (copied average of pretrained RGB weights).
"""

import sys
from pathlib import Path
import torch
import timm

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.swin_model import SwinBackbone, adapt_input_stem, build_swin_backbone


def run_swin_model_test():
    print("=" * 70)
    print("=== PRETRAINED SWIN TRANSFORMER & STEM ADAPTATION TEST ===")
    print("=" * 70)

    backbone_name = "swin_tiny_patch4_window7_224"
    batch_size = 2
    image_size = (224, 224)

    # Step 1: Instantiate standard 3-channel model (Kaggle setting)
    print(f"\n[1] Testing Standard 3-Channel Swin Model ({backbone_name})...")
    model_3ch = SwinBackbone(
        backbone_name=backbone_name,
        input_channels=3,
        pretrained=True,
    )

    print(f"    Initial Parameter Count (3ch): {model_3ch.initial_param_count:,}")
    print(f"    Adapted Parameter Count (3ch): {model_3ch.num_parameters:,}")
    print(f"    Feature Embedding Dimension:   {model_3ch.num_features}")
    assert model_3ch.initial_param_count == model_3ch.num_parameters, "3-channel model params should not change!"

    # Forward pass with 3-channel dummy tensor
    dummy_3ch = torch.randn(batch_size, 3, *image_size, dtype=torch.float32)
    feat_3ch = model_3ch(dummy_3ch)
    
    print(f"    Input Dummy Tensor Shape:  {list(dummy_3ch.shape)}")
    print(f"    Output Feature Map Shape:  {list(feat_3ch.shape)} (Expected: [{batch_size}, 768])")
    print(f"    Output Data Type:          {feat_3ch.dtype}")
    
    assert feat_3ch.shape == (batch_size, 768), f"Unexpected shape {feat_3ch.shape}"
    assert not torch.isnan(feat_3ch).any(), "NaN found in 3ch output features!"
    assert not torch.isinf(feat_3ch).any(), "Inf found in 3ch output features!"
    print("    -> 3-Channel Swin Backbone output verified successfully.")

    # Step 2: Instantiate adapted 4-channel model (BraTS Early Fusion setting)
    print(f"\n[2] Testing Adapted 4-Channel Swin Model ({backbone_name})...")
    
    # Measure baseline raw timm model 3ch stem weight
    raw_3ch_timm = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    old_proj_weight = raw_3ch_timm.patch_embed.proj.weight.clone().detach()  # [96, 3, 4, 4]

    model_4ch = SwinBackbone(
        backbone_name=backbone_name,
        input_channels=4,
        pretrained=True,
    )

    print(f"    Parameter Count BEFORE stem adaptation (3ch baseline): {model_4ch.initial_param_count:,}")
    print(f"    Parameter Count AFTER stem adaptation (4ch adapted):   {model_4ch.num_parameters:,}")
    print(f"    Net Parameter Increase (due to 4th channel stem):     +{model_4ch.num_parameters - model_4ch.initial_param_count:,} parameters")

    new_proj_weight = model_4ch.backbone.patch_embed.proj.weight.detach()  # [96, 4, 4, 4]
    print(f"    Old Stem Conv Weight Shape: {list(old_proj_weight.shape)}")
    print(f"    New Stem Conv Weight Shape: {list(new_proj_weight.shape)}")

    # Verify weight initialization math: Channels 0..2 match old weights, Channel 3 equals average of RGB weights
    print("\n[3] Verifying Weight Inflation Rationale & Weight Copying...")
    rgb_copy_diff = (new_proj_weight[:, :3, :, :] - old_proj_weight).abs().max().item()
    expected_mean = old_proj_weight.mean(dim=1, keepdim=True)
    ch4_mean_diff = (new_proj_weight[:, 3:4, :, :] - expected_mean).abs().max().item()

    print(f"    Max Absolute Diff (Channels 0-2 vs Old Pretrained Weights): {rgb_copy_diff:.6f}")
    print(f"    Max Absolute Diff (Channel 3 vs Mean RGB Weights):          {ch4_mean_diff:.6f}")

    assert rgb_copy_diff < 1e-6, "RGB pretrained weights were not accurately copied!"
    assert ch4_mean_diff < 1e-6, "4th channel weight initialization does not match RGB mean!"
    print("    -> Weight inflation math verified 100% accurate.")

    # Forward pass with 4-channel dummy tensor
    print("\n[4] Performing Forward Pass on 4-Channel Input...")
    dummy_4ch = torch.randn(batch_size, 4, *image_size, dtype=torch.float32)
    feat_4ch = model_4ch(dummy_4ch)

    print(f"    Input Dummy Tensor Shape:  {list(dummy_4ch.shape)}")
    print(f"    Output Feature Map Shape:  {list(feat_4ch.shape)} (Expected: [{batch_size}, 768])")
    print(f"    Output Data Type:          {feat_4ch.dtype}")

    assert feat_4ch.shape == (batch_size, 768), f"Unexpected shape {feat_4ch.shape}"
    assert not torch.isnan(feat_4ch).any(), "NaN found in 4ch output features!"
    assert not torch.isinf(feat_4ch).any(), "Inf found in 4ch output features!"
    print("    -> 4-Channel Swin Backbone output verified successfully.")

    # Step 5: Test Factory Function build_swin_backbone
    print("\n[5] Testing Factory Function 'build_swin_backbone'...")
    factory_model = build_swin_backbone(input_channels=4, pretrained=True)
    factory_feat = factory_model(dummy_4ch)
    assert factory_feat.shape == (batch_size, 768)
    print("    -> Factory function verified working cleanly.")

    print("\n" + "=" * 70)
    print("=== ALL SWIN BACKBONE & STEM ADAPTATION TESTS PASSED! ===")
    print("=" * 70)


if __name__ == "__main__":
    run_swin_model_test()
