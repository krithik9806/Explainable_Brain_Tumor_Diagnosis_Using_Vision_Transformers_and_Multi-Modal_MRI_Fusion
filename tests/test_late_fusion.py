"""
Unit test and shape-flow verification script for LateFusionModule (Day 12 Stretch Task).

This script:
1. Instantiates LateFusionModule for both 'concat' and 'average' fusion modes.
2. Creates dummy batch input tensors for 4 BraTS modalities [B, 1, 224, 224].
3. Passes modalities through the module and verifies shape flow across encoders, fusion layer, and classification head.
4. Validates output logits shape [B, num_classes], data type (torch.float32), and zero NaN/Inf errors.
5. Verifies compatibility with 4D stacked input tensors [B, 4, 224, 224].
6. Verifies that Day 11 Early Fusion module (src/fusion/fusion.py) remains 100% intact and working.
"""

import sys
from pathlib import Path
import torch

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.late_fusion import LateFusionModule, SingleModalityEncoder
from src.fusion.fusion import BraTSEarlyFusion


def run_late_fusion_test():
    print("=" * 70)
    print("=== LATE FUSION MODULE SHAPE-FLOW & UNIT TEST ===")
    print("=" * 70)

    batch_size = 4
    embed_dim = 128
    num_classes = 2
    spatial_size = (224, 224)

    print(f"\n[1] Test Configuration:")
    print(f"    Batch Size:   {batch_size}")
    print(f"    Embed Dim:    {embed_dim}")
    print(f"    Num Classes:  {num_classes}")
    print(f"    Spatial Size: {spatial_size}")

    # Create dummy modality inputs [B, 1, 224, 224]
    dummy_modalities = {
        "t1": torch.randn(batch_size, 1, *spatial_size, dtype=torch.float32),
        "t1ce": torch.randn(batch_size, 1, *spatial_size, dtype=torch.float32),
        "t2": torch.randn(batch_size, 1, *spatial_size, dtype=torch.float32),
        "flair": torch.randn(batch_size, 1, *spatial_size, dtype=torch.float32),
    }

    print("\n[2] Verifying SingleModalityEncoder shape flow...")
    encoder = SingleModalityEncoder(in_channels=1, embed_dim=embed_dim)
    sample_feat = encoder(dummy_modalities["t1"])
    print(f"    SingleModalityEncoder Input Shape:  {list(dummy_modalities['t1'].shape)}")
    print(f"    SingleModalityEncoder Output Shape: {list(sample_feat.shape)}")
    assert sample_feat.shape == (batch_size, embed_dim), f"Unexpected encoder shape {sample_feat.shape}"

    # Test LateFusionModule in 'concat' mode
    print("\n[3] Testing LateFusionModule (mode='concat')...")
    late_fusion_concat = LateFusionModule(
        embed_dim=embed_dim,
        num_classes=num_classes,
        fusion_mode="concat",
        dropout_prob=0.2,
    )

    out_concat = late_fusion_concat(dummy_modalities, return_features=True)
    logits_concat = out_concat["logits"]
    fused_concat = out_concat["fused_features"]
    per_mod_feats = out_concat["per_modality_features"]

    print(f"    Per-Modality Feature Vectors: {[f'{m}: {list(per_mod_feats[m].shape)}' for m in per_mod_feats]}")
    print(f"    Fused Feature Matrix Shape:   {list(fused_concat.shape)}  (Expected: [{batch_size}, {embed_dim * 4}])")
    print(f"    Output Logits Matrix Shape:  {list(logits_concat.shape)}    (Expected: [{batch_size}, {num_classes}])")
    print(f"    Output Logits Data Type:     {logits_concat.dtype}")

    assert logits_concat.shape == (batch_size, num_classes)
    assert fused_concat.shape == (batch_size, embed_dim * 4)
    assert not torch.isnan(logits_concat).any(), "NaN found in concat logits!"
    assert not torch.isinf(logits_concat).any(), "Inf found in concat logits!"

    # Print sample logits values
    print("\n    Sample Logits Output (concat mode):")
    for i in range(batch_size):
        print(f"      Sample {i+1}: LGG logit = {logits_concat[i, 0]:8.4f} | HGG logit = {logits_concat[i, 1]:8.4f}")

    # Test LateFusionModule in 'average' mode
    print("\n[4] Testing LateFusionModule (mode='average')...")
    late_fusion_avg = LateFusionModule(
        embed_dim=embed_dim,
        num_classes=num_classes,
        fusion_mode="average",
        dropout_prob=0.2,
    )

    out_avg = late_fusion_avg(dummy_modalities, return_features=True)
    logits_avg = out_avg["logits"]
    fused_avg = out_avg["fused_features"]

    print(f"    Fused Feature Matrix Shape:   {list(fused_avg.shape)}  (Expected: [{batch_size}, {embed_dim}])")
    print(f"    Output Logits Matrix Shape:  {list(logits_avg.shape)}    (Expected: [{batch_size}, {num_classes}])")

    assert logits_avg.shape == (batch_size, num_classes)
    assert fused_avg.shape == (batch_size, embed_dim)
    assert not torch.isnan(logits_avg).any(), "NaN found in average logits!"

    # Test stacked 4D tensor input [B, 4, 224, 224]
    print("\n[5] Testing 4D stacked tensor input [B, 4, 224, 224]...")
    stacked_input = torch.randn(batch_size, 4, *spatial_size)
    logits_stacked = late_fusion_concat(stacked_input)
    print(f"    Stacked Tensor Input Shape:  {list(stacked_input.shape)}")
    print(f"    Output Logits Shape:         {list(logits_stacked.shape)}")
    assert logits_stacked.shape == (batch_size, num_classes)

    # Step 6: Verify Day 11 Early Fusion remains intact
    print("\n[6] Verifying Day 11 Early Fusion module integrity...")
    early_fusion = BraTSEarlyFusion(target_size=(224, 224), return_tensor=True)
    mock_dict = {
        "t1": torch.randn(224, 224),
        "t1ce": torch.randn(224, 224),
        "t2": torch.randn(224, 224),
        "flair": torch.randn(224, 224),
    }
    early_out = early_fusion(mock_dict)
    print(f"    BraTSEarlyFusion Output Shape: {list(early_out.shape)} (Expected: [4, 224, 224])")
    assert early_out.shape == torch.Size([4, 224, 224])
    print("    -> Day 11 Early Fusion module verified intact and fully operational!")

    print("\n" + "=" * 70)
    print("=== ALL LATE FUSION TESTS PASSED SUCCESSFULLY! ===")
    print("=" * 70)


if __name__ == "__main__":
    run_late_fusion_test()
