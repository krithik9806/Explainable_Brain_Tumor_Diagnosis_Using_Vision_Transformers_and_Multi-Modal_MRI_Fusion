"""
End-to-End Pipeline Smoke Test Script (Day 15 Task).

This script verifies end-to-end integration between:
1. Dataset & DataLoader classes (src/data/datasets.py).
2. Multi-Modal Fusion module (src/fusion/fusion.py).
3. Pretrained Swin Transformer & Classification Head (src/models/swin_model.py).

Test Flow:
- Instantiates DataLoaders for train, val, and test splits for Kaggle and BraTS datasets.
- Fetches 1 real batch from Kaggle train loader [batch_size, 3, 224, 224] and label tensor [batch_size].
- Passes batch through SwinClassifier(input_channels=3, num_classes=4) and verifies output logits shape [batch_size, 4].
- Fetches 1 real batch from BraTS train loader [batch_size, 4, 224, 224] and label tensor [batch_size].
- Passes batch through SwinClassifier(input_channels=4, num_classes=2) and verifies output logits shape [batch_size, 2].
- Verifies zero NaNs, Infs, or shape errors.
"""

import sys
import traceback
from pathlib import Path
import torch
import yaml

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import (
    BraTSDataset,
    KaggleDataset,
    get_brats_dataloaders,
    get_kaggle_dataloaders,
    get_dataloaders_from_config,
)
from src.models.swin_model import SwinClassifier, build_swin_classifier


def run_pipeline_smoketest():
    print("=" * 75)
    print("=== END-TO-END PIPELINE INTEGRATION SMOKE TEST ===")
    print("=" * 75)

    try:
        # Load configs
        kaggle_cfg_path = PROJECT_ROOT / "configs" / "kaggle_config.yaml"
        brats_cfg_path = PROJECT_ROOT / "configs" / "brats_fusion_config.yaml"

        with open(kaggle_cfg_path, "r") as f:
            kaggle_cfg = yaml.safe_load(f)
        with open(brats_cfg_path, "r") as f:
            brats_cfg = yaml.safe_load(f)

        kaggle_batch_size = kaggle_cfg["training"]["batch_size"]  # 32
        brats_batch_size = brats_cfg["training"]["batch_size"]    # 16

        print("\n[1] Testing Kaggle DataLoaders & SwinClassifier Pipeline...")
        kaggle_loaders = get_kaggle_dataloaders(batch_size=kaggle_batch_size)
        
        for split in ["train", "val", "test"]:
            loader = kaggle_loaders[split]
            dataset_size = len(loader.dataset)
            num_batches = len(loader)
            print(f"    Kaggle '{split:<5}' DataLoader: {dataset_size:,} samples | {num_batches} batches (batch_size={loader.batch_size})")
            assert dataset_size > 0, f"Kaggle {split} dataset is empty!"

        # Pull 1 real batch from Kaggle train loader
        kaggle_train_iter = iter(kaggle_loaders["train"])
        kaggle_images, kaggle_labels = next(kaggle_train_iter)

        print(f"\n    Real Kaggle Batch Loaded:")
        print(f"      Image Batch Shape:  {list(kaggle_images.shape)} (Expected: [{kaggle_batch_size}, 3, 224, 224])")
        print(f"      Image Batch Dtype:  {kaggle_images.dtype}")
        print(f"      Label Batch Shape:  {list(kaggle_labels.shape)} (Expected: [{kaggle_batch_size}])")
        print(f"      Label Batch Dtype:  {kaggle_labels.dtype}")
        print(f"      Sample Label Indices: {kaggle_labels[:8].tolist()}")

        assert kaggle_images.shape == (kaggle_batch_size, 3, 224, 224)
        assert kaggle_labels.shape == (kaggle_batch_size,)

        # Instantiate Kaggle Classifier
        print("\n    Instantiating Kaggle SwinClassifier (3 channels, 4 classes)...")
        kaggle_model = build_swin_classifier(
            input_channels=3,
            num_classes=4,
            pretrained=True,
        )

        with torch.no_grad():
            kaggle_logits = kaggle_model(kaggle_images)

        print(f"    Output Logits Shape: {list(kaggle_logits.shape)} (Expected: [{kaggle_batch_size}, 4])")
        assert kaggle_logits.shape == (kaggle_batch_size, 4)
        assert not torch.isnan(kaggle_logits).any(), "NaN found in Kaggle logits!"
        assert not torch.isinf(kaggle_logits).any(), "Inf found in Kaggle logits!"
        print("    -> Kaggle end-to-end pipeline smoke test PASSED!")

        # Step 2: Test BraTS DataLoaders & Classifier Pipeline
        print("\n[2] Testing BraTS DataLoaders & SwinClassifier Pipeline...")
        brats_loaders = get_brats_dataloaders(batch_size=brats_batch_size)

        for split in ["train", "val", "test"]:
            loader = brats_loaders[split]
            dataset_size = len(loader.dataset)
            num_batches = len(loader)
            print(f"    BraTS  '{split:<5}' DataLoader: {dataset_size:,} samples | {num_batches} batches (batch_size={loader.batch_size})")
            assert dataset_size > 0, f"BraTS {split} dataset is empty!"

        # Pull 1 real batch from BraTS train loader
        brats_train_iter = iter(brats_loaders["train"])
        brats_images, brats_labels = next(brats_train_iter)

        print(f"\n    Real BraTS Batch Loaded:")
        print(f"      Image Batch Shape:  {list(brats_images.shape)} (Expected: [{brats_batch_size}, 4, 224, 224])")
        print(f"      Image Batch Dtype:  {brats_images.dtype}")
        print(f"      Label Batch Shape:  {list(brats_labels.shape)} (Expected: [{brats_batch_size}])")
        print(f"      Label Batch Dtype:  {brats_labels.dtype}")
        print(f"      Sample Label Indices: {brats_labels[:8].tolist()}")

        assert brats_images.shape == (brats_batch_size, 4, 224, 224)
        assert brats_labels.shape == (brats_batch_size,)

        # Instantiate BraTS Classifier
        print("\n    Instantiating BraTS SwinClassifier (4 channels, 2 classes)...")
        brats_model = build_swin_classifier(
            input_channels=4,
            num_classes=2,
            pretrained=True,
        )

        with torch.no_grad():
            brats_logits = brats_model(brats_images)

        print(f"    Output Logits Shape: {list(brats_logits.shape)} (Expected: [{brats_batch_size}, 2])")
        assert brats_logits.shape == (brats_batch_size, 2)
        assert not torch.isnan(brats_logits).any(), "NaN found in BraTS logits!"
        assert not torch.isinf(brats_logits).any(), "Inf found in BraTS logits!"
        print("    -> BraTS end-to-end pipeline smoke test PASSED!")

        # Step 3: Test Factory get_dataloaders_from_config
        print("\n[3] Testing 'get_dataloaders_from_config' helper...")
        config_loaders_b = get_dataloaders_from_config(brats_cfg_path)
        assert "train" in config_loaders_b and "val" in config_loaders_b and "test" in config_loaders_b
        print("    -> Config loader helper successfully instantiated all split DataLoaders.")

        print("\n" + "=" * 75)
        print("=== END-TO-END PIPELINE SMOKE TEST PASSED WITH ZERO ERRORS! ===")
        print("=" * 75)

    except Exception as e:
        print("\n" + "!" * 75)
        print("!!! PIPELINE SMOKE TEST FAILED !!!")
        print(f"Error Type: {type(e).__name__}: {e}")
        print("Full Traceback:")
        traceback.print_exc()
        print("!" * 75)
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline_smoketest()
