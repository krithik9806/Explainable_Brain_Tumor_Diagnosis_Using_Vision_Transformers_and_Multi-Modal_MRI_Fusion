"""
Unit test and verification script for SwinClassifier and Custom Classification Head (Day 14 Task).

This script:
1. Loads configs (configs/kaggle_config.yaml and configs/brats_fusion_config.yaml) to verify num_classes and input_channels.
2. Instantiates SwinClassifier for Kaggle configuration (3 channels, 4 classes).
   Passes dummy batch [2, 3, 224, 224] and verifies output logits shape [2, 4].
3. Instantiates SwinClassifier for BraTS configuration (4 channels, 2 classes).
   Passes dummy batch [2, 4, 224, 224] and verifies output logits shape [2, 2].
4. Applies torch.softmax to logits and verifies probabilities sum to 1.0 per sample.
5. Verifies absence of NaNs or shape mismatches across all configurations.
"""

import sys
from pathlib import Path
import torch
import yaml

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.swin_model import SwinClassifier, build_swin_classifier


def run_swin_classifier_test():
    print("=" * 70)
    print("=== SWIN CLASSIFIER & CUSTOM HEAD VERIFICATION TEST ===")
    print("=" * 70)

    # Step 1: Read configs to verify setup
    kaggle_cfg_path = PROJECT_ROOT / "configs" / "kaggle_config.yaml"
    brats_cfg_path = PROJECT_ROOT / "configs" / "brats_fusion_config.yaml"

    with open(kaggle_cfg_path, "r") as f:
        kaggle_cfg = yaml.safe_load(f)
    with open(brats_cfg_path, "r") as f:
        brats_cfg = yaml.safe_load(f)

    kaggle_in_chans = kaggle_cfg["dataset"]["input_channels"]
    kaggle_num_classes = kaggle_cfg["dataset"]["num_classes"]
    kaggle_class_names = kaggle_cfg["dataset"]["class_names"]

    brats_in_chans = brats_cfg["dataset"]["input_channels"]
    brats_num_classes = brats_cfg["dataset"]["num_classes"]
    brats_class_names = brats_cfg["dataset"]["class_names"]

    print(f"\n[1] Configuration Check:")
    print(f"    Kaggle Config: input_channels={kaggle_in_chans}, num_classes={kaggle_num_classes}, class_names={kaggle_class_names}")
    print(f"    BraTS Config:  input_channels={brats_in_chans}, num_classes={brats_num_classes}, class_names={brats_class_names}")

    # Step 2: Test Kaggle Model (3 Channels, 4 Classes)
    print(f"\n[2] Testing Kaggle SwinClassifier Setup (3 channels, 4 classes)...")
    batch_size = 2
    model_kaggle = SwinClassifier(
        backbone_name="swin_tiny_patch4_window7_224",
        input_channels=kaggle_in_chans,
        num_classes=kaggle_num_classes,
        pretrained=True,
        drop_rate=0.2,
    )

    print(f"    Total Model Parameters:       {model_kaggle.num_parameters:,}")
    print(f"    Backbone Feature Dimension:   {model_kaggle.num_features}")

    dummy_kaggle = torch.randn(batch_size, kaggle_in_chans, 224, 224, dtype=torch.float32)
    logits_kaggle, feats_kaggle = model_kaggle(dummy_kaggle, return_features=True)

    print(f"    Input Dummy Batch Shape:      {list(dummy_kaggle.shape)}")
    print(f"    Extracted Feature Shape:      {list(feats_kaggle.shape)} (Expected: [{batch_size}, 768])")
    print(f"    Output Logits Shape:          {list(logits_kaggle.shape)} (Expected: [{batch_size}, 4])")
    print(f"    Logits Data Type:             {logits_kaggle.dtype}")

    assert logits_kaggle.shape == (batch_size, 4), f"Unexpected Kaggle logits shape {logits_kaggle.shape}"
    assert not torch.isnan(logits_kaggle).any(), "NaN found in Kaggle logits!"
    assert not torch.isinf(logits_kaggle).any(), "Inf found in Kaggle logits!"

    # Apply softmax and check probability sums
    probs_kaggle = torch.softmax(logits_kaggle, dim=-1)
    prob_sums_kaggle = probs_kaggle.sum(dim=-1)
    print(f"    Softmax Probabilities Matrix:\n{probs_kaggle.detach().cpu().numpy()}")
    print(f"    Probability Sums per Sample: {prob_sums_kaggle.tolist()}")
    for sum_val in prob_sums_kaggle:
        assert abs(sum_val.item() - 1.0) < 1e-5, f"Probability sum {sum_val.item()} is not 1.0!"
    print("    -> Kaggle 4-class classifier verified successfully.")

    # Step 3: Test BraTS Early Fusion Model (4 Channels, 2 Classes)
    print(f"\n[3] Testing BraTS SwinClassifier Setup (4 channels, 2 classes)...")
    model_brats = SwinClassifier(
        backbone_name="swin_tiny_patch4_window7_224",
        input_channels=brats_in_chans,
        num_classes=brats_num_classes,
        pretrained=True,
        drop_rate=0.2,
    )

    print(f"    Total Model Parameters:       {model_brats.num_parameters:,}")
    print(f"    Backbone Feature Dimension:   {model_brats.num_features}")

    dummy_brats = torch.randn(batch_size, brats_in_chans, 224, 224, dtype=torch.float32)
    logits_brats, feats_brats = model_brats(dummy_brats, return_features=True)

    print(f"    Input Dummy Batch Shape:      {list(dummy_brats.shape)}")
    print(f"    Extracted Feature Shape:      {list(feats_brats.shape)} (Expected: [{batch_size}, 768])")
    print(f"    Output Logits Shape:          {list(logits_brats.shape)} (Expected: [{batch_size}, 2])")

    assert logits_brats.shape == (batch_size, 2), f"Unexpected BraTS logits shape {logits_brats.shape}"
    assert not torch.isnan(logits_brats).any(), "NaN found in BraTS logits!"
    assert not torch.isinf(logits_brats).any(), "Inf found in BraTS logits!"

    # Apply softmax and check probability sums
    probs_brats = torch.softmax(logits_brats, dim=-1)
    prob_sums_brats = probs_brats.sum(dim=-1)
    print(f"    Softmax Probabilities Matrix:\n{probs_brats.detach().cpu().numpy()}")
    print(f"    Probability Sums per Sample: {prob_sums_brats.tolist()}")
    for sum_val in prob_sums_brats:
        assert abs(sum_val.item() - 1.0) < 1e-5, f"Probability sum {sum_val.item()} is not 1.0!"
    print("    -> BraTS 2-class classifier verified successfully.")

    # Step 4: Test Factory Function build_swin_classifier
    print("\n[4] Testing Factory Function 'build_swin_classifier'...")
    factory_classifier = build_swin_classifier(input_channels=4, num_classes=2, pretrained=True)
    factory_logits = factory_classifier(dummy_brats)
    assert factory_logits.shape == (batch_size, 2)
    print("    -> Factory function verified working cleanly.")

    print("\n" + "=" * 70)
    print("=== ALL SWIN CLASSIFIER TESTS PASSED SUCCESSFULLY! ===")
    print("=" * 70)


if __name__ == "__main__":
    run_swin_classifier_test()
