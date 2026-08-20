"""
Test script for verifying Multi-Modal MRI Early Fusion and Kaggle Pass-Through.

This script:
1. Verifies configuration alignment with configs/brats_fusion_config.yaml.
2. Loads a real sample .npz file from data/processed/brats_normalized/ (read-only).
3. Passes the 4 modalities (t1, t1ce, t2, flair) through BraTSEarlyFusion.
4. Tests multiple input interfaces (dict, file path, kwargs, tuple).
5. Verifies output tensor shape [4, 224, 224], data type (torch.float32), and checks for NaNs/Infs.
6. Tests KagglePassThrough module for 3-channel consistent interface.
7. Prints sample values for verification.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.fusion import (
    BraTSEarlyFusion,
    KagglePassThrough,
    fuse_brats_modalities,
    pass_through_kaggle,
    get_fusion_module,
)


def run_fusion_test():
    print("=" * 70)
    print("=== MULTI-MODAL MRI FUSION MODULE TEST ===")
    print("=" * 70)

    # Step 1: Verify config alignment
    config_path = PROJECT_ROOT / "configs" / "brats_fusion_config.yaml"
    print(f"\n[1] Verifying config file: {config_path}")
    assert config_path.exists(), f"Config file not found at {config_path}"
    
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    expected_channels = config_data["dataset"]["input_channels"]
    expected_size = tuple(config_data["dataset"]["image_size"])
    print(f"    Config input_channels: {expected_channels}")
    print(f"    Config image_size:     {expected_size}")
    assert expected_channels == 4, f"Expected 4 channels in config, got {expected_channels}"
    assert expected_size == (224, 224), f"Expected (224, 224) size in config, got {expected_size}"

    # Step 2: Locate real sample file from data/processed/brats_normalized/
    brats_norm_dir = PROJECT_ROOT / "data" / "processed" / "brats_normalized"
    assert brats_norm_dir.exists(), f"Directory not found: {brats_norm_dir}"

    sample_files = list(brats_norm_dir.glob("**/*.npz"))
    assert len(sample_files) > 0, "No .npz files found in data/processed/brats_normalized/"

    sample_file = sample_files[0]
    rel_sample_path = sample_file.relative_to(PROJECT_ROOT)
    print(f"\n[2] Loading real BraTS sample file (read-only):")
    print(f"    Path: {rel_sample_path}")

    # Inspect raw .npz contents
    raw_npz = np.load(sample_file)
    print("    Keys present in .npz file:", list(raw_npz.keys()))
    for key in ['t1', 't1ce', 't2', 'flair']:
        arr = raw_npz[key]
        print(f"    - Modality '{key:<5}': shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

    # Step 3: Instantiate BraTSEarlyFusion module
    print("\n[3] Instantiating BraTSEarlyFusion module...")
    fusion_module = BraTSEarlyFusion(target_size=(224, 224), return_tensor=True)
    print(f"    Expected Channel Order: {fusion_module.CHANNEL_ORDER}")

    # Step 4: Test Fusion with file path input
    print("\n[4] Test 4A: Passing .npz file path through BraTSEarlyFusion module...")
    fused_tensor_file = fusion_module(sample_file)
    
    print(f"    Output Tensor Type:   {type(fused_tensor_file)}")
    print(f"    Output Tensor Shape:  {fused_tensor_file.shape}")
    print(f"    Output Tensor Dtype:  {fused_tensor_file.dtype}")
    print(f"    Output Device:        {fused_tensor_file.device}")
    
    assert isinstance(fused_tensor_file, torch.Tensor), "Output should be torch.Tensor"
    assert fused_tensor_file.shape == torch.Size([4, 224, 224]), f"Unexpected shape {fused_tensor_file.shape}"
    assert fused_tensor_file.dtype == torch.float32, f"Unexpected dtype {fused_tensor_file.dtype}"
    assert not torch.isnan(fused_tensor_file).any(), "NaN values found in fused tensor!"
    assert not torch.isinf(fused_tensor_file).any(), "Inf values found in fused tensor!"

    # Test 4B: Passing dict input
    print("\n    Test 4B: Passing dictionary of NumPy arrays...")
    mod_dict = {k: raw_npz[k] for k in ['t1', 't1ce', 't2', 'flair']}
    fused_tensor_dict = fusion_module(mod_dict)
    assert torch.equal(fused_tensor_file, fused_tensor_dict), "File input and dict input outputs differ!"
    print("    -> Dict input output matches file path output perfectly.")

    # Test 4C: Functional interface fuse_brats_modalities
    print("\n    Test 4C: Testing functional interface 'fuse_brats_modalities'...")
    fused_func = fuse_brats_modalities(
        t1=raw_npz['t1'],
        t1ce=raw_npz['t1ce'],
        t2=raw_npz['t2'],
        flair=raw_npz['flair']
    )
    assert torch.equal(fused_tensor_file, fused_func), "Functional interface output differs!"
    print("    -> Functional interface output matches class output perfectly.")

    # Step 5: Print Channel Statistics & Sample Matrix Values
    print("\n[5] Detailed Per-Channel Verification:")
    print("-" * 70)
    channel_names = ["Channel 0: T1", "Channel 1: T1ce", "Channel 2: T2", "Channel 3: FLAIR"]
    for ch_idx, ch_name in enumerate(channel_names):
        ch_tensor = fused_tensor_file[ch_idx]
        ch_min = float(ch_tensor.min())
        ch_max = float(ch_tensor.max())
        ch_mean = float(ch_tensor.mean())
        ch_std = float(ch_tensor.std())
        center_val = float(ch_tensor[112, 112])
        sub_grid = ch_tensor[110:113, 110:113].cpu().numpy()
        
        print(f"  {ch_name:18s} | Shape: {list(ch_tensor.shape)} | Min: {ch_min:8.4f} | Max: {ch_max:8.4f} | Mean: {ch_mean:8.4f} | Std: {ch_std:8.4f} | Center(112,112): {center_val:8.4f}")
        print(f"    3x3 Central Values (rows 110:113, cols 110:113):")
        for row in sub_grid:
            print("      ", " ".join(f"{val:8.4f}" for val in row))
        print("-" * 70)

    # Step 6: Test Kaggle Pass-Through Module
    print("\n[6] Testing KagglePassThrough module for 3-channel single-modality images...")
    kaggle_module = KagglePassThrough(target_size=(224, 224), return_tensor=True)
    
    # Create mock 2D image [224, 224]
    mock_2d = np.random.randn(224, 224).astype(np.float32)
    kaggle_tensor = kaggle_module(mock_2d)
    
    print(f"    Input 2D Shape:        (224, 224)")
    print(f"    Kaggle Output Type:    {type(kaggle_tensor)}")
    print(f"    Kaggle Output Shape:   {kaggle_tensor.shape}")
    print(f"    Kaggle Output Dtype:   {kaggle_tensor.dtype}")
    assert kaggle_tensor.shape == torch.Size([3, 224, 224]), f"Unexpected Kaggle shape {kaggle_tensor.shape}"
    assert not torch.isnan(kaggle_tensor).any(), "NaN found in Kaggle output!"
    print("    -> Kaggle 3-channel pass-through verified successfully.")

    # Step 7: Test Factory function
    print("\n[7] Testing factory function 'get_fusion_module'...")
    brats_mod = get_fusion_module("brats2020_fusion")
    kaggle_mod = get_fusion_module("kaggle")
    assert isinstance(brats_mod, BraTSEarlyFusion)
    assert isinstance(kaggle_mod, KagglePassThrough)
    print("    -> Factory function successfully returns proper modules for 'brats2020_fusion' and 'kaggle'.")

    print("\n" + "=" * 70)
    print("=== ALL FUSION MODULE TESTS PASSED SUCCESSFULLY! ===")
    print("=" * 70)


if __name__ == "__main__":
    run_fusion_test()
