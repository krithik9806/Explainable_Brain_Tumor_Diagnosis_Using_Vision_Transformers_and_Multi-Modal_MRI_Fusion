"""
Patient-Level Dataset Splitting & Data Leakage Prevention Module.

This script performs reproducible train/validation/test dataset splits:
1. BraTS Dataset:
   - Patient-level splitting (70% train / 15% val / 15% test) to prevent slice-level data leakage.
   - Stratified by tumor grade (HGG vs LGG).
   - Fixed random seed (seed=42) from configs/base_config.yaml.
   - Saves mapping to `data/processed/brats_splits.csv`.
   - Programmatically verifies zero patient ID overlap across splits.

2. Kaggle Dataset:
   - Preserves official `Testing` folder as held-out test set (1,600 images).
   - Carves 20% validation set out of `Training` folder (1,120 val images, 4,480 train images).
   - Stratified across 4 classes (glioma, meningioma, notumor, pituitary).
   - Saves mapping to `data/processed/kaggle_splits.csv`.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_brats_dataset(
    manifest_path: Path,
    output_csv: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, bool, Dict]:
    """
    Split BraTS dataset at the unique patient level into train/val/test splits.

    Returns:
        Tuple containing:
        - Full DataFrame with assigned splits.
        - Boolean indicating whether patient leakage check PASSED (True) or FAILED (False).
        - Summary metrics dictionary.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"BraTS manifest file not found at {manifest_path}")

    df_manifest = pd.read_csv(manifest_path)
    
    # Extract unique patient IDs and their tumor grade (HGG / LGG)
    patient_df = df_manifest[['patient_id', 'grade']].drop_duplicates().reset_index(drop=True)
    
    # First split: Train (70%) vs Temp (30%)
    temp_ratio = val_ratio + test_ratio  # 0.30
    train_patients, temp_patients = train_test_split(
        patient_df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=patient_df['grade'],
    )

    # Second split: Val (15%) vs Test (15%) -> 50/50 split of Temp
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.5,
        random_state=seed,
        stratify=temp_patients['grade'],
    )

    # Build patient -> split map
    patient_split_map = {}
    for p_id in train_patients['patient_id']:
        patient_split_map[p_id] = 'train'
    for p_id in val_patients['patient_id']:
        patient_split_map[p_id] = 'val'
    for p_id in test_patients['patient_id']:
        patient_split_map[p_id] = 'test'

    # Assign split to every slice in manifest
    df_manifest['split'] = df_manifest['patient_id'].map(patient_split_map)
    
    # Update file paths to point to normalized directory
    df_manifest['file_path'] = df_manifest['file_path'].apply(
        lambda p: str(Path(p).as_posix()).replace("data/processed/brats_slices", "data/processed/brats_normalized")
    )

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_manifest.to_csv(output_csv, index=False)

    # Verification: Check patient-level leakage across splits
    train_set = set(train_patients['patient_id'])
    val_set = set(val_patients['patient_id'])
    test_set = set(test_patients['patient_id'])

    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)

    has_no_leakage = (len(overlap_train_val) == 0 and 
                      len(overlap_train_test) == 0 and 
                      len(overlap_val_test) == 0)

    summary = {
        "unique_patients": len(patient_df),
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
        "test_patients": len(test_patients),
        "total_slices": len(df_manifest),
        "train_slices": len(df_manifest[df_manifest['split'] == 'train']),
        "val_slices": len(df_manifest[df_manifest['split'] == 'val']),
        "test_slices": len(df_manifest[df_manifest['split'] == 'test']),
        "leakage_pass": has_no_leakage,
    }

    return df_manifest, has_no_leakage, summary


def split_kaggle_dataset(
    kaggle_dir: Path,
    output_csv: Path,
    seed: int = 42,
    val_ratio_of_training: float = 0.20,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Split Kaggle dataset into train/val/test splits.

    Preserves existing `Testing/` folder as held-out test split,
    and carves 20% validation set out of `Training/` folder, stratified by class.
    """
    if not kaggle_dir.exists():
        raise FileNotFoundError(f"Kaggle directory not found at {kaggle_dir}")

    # Gather all normalized .npz files (or raw images)
    all_files = sorted(list(kaggle_dir.glob("**/*.npz")))
    if not all_files:
        all_files = sorted(list(kaggle_dir.glob("**/*.jpg")) + list(kaggle_dir.glob("**/*.png")))

    records = []
    for f in all_files:
        rel_parts = f.relative_to(kaggle_dir).parts
        orig_folder = rel_parts[0]  # 'Training' or 'Testing'
        class_name = rel_parts[1]   # 'glioma', 'meningioma', 'notumor', 'pituitary'
        filename = f.name
        
        records.append({
            "filename": filename,
            "class_name": class_name,
            "original_folder": orig_folder,
            "file_path": str(f.relative_to(kaggle_dir.parents[1]).as_posix()),
        })

    df = pd.DataFrame(records)

    # Testing folder -> 'test' split
    test_df = df[df['original_folder'] == 'Testing'].copy()
    test_df['split'] = 'test'

    # Training folder -> stratify into 'train' (80%) and 'val' (20%)
    training_df = df[df['original_folder'] == 'Training'].copy()
    train_sub, val_sub = train_test_split(
        training_df,
        test_size=val_ratio_of_training,
        random_state=seed,
        stratify=training_df['class_name'],
    )

    train_sub['split'] = 'train'
    val_sub['split'] = 'val'

    final_df = pd.concat([train_sub, val_sub, test_df]).reset_index(drop=True)

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    summary = {
        "total_images": len(final_df),
        "train_images": len(final_df[final_df['split'] == 'train']),
        "val_images": len(final_df[final_df['split'] == 'val']),
        "test_images": len(final_df[final_df['split'] == 'test']),
    }

    return final_df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Dataset splitting script for BraTS and Kaggle brain MRI data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--brats_manifest",
        type=str,
        default="data/processed/brats_normalized/manifest.csv",
        help="Path to BraTS manifest CSV file.",
    )
    parser.add_argument(
        "--brats_out_csv",
        type=str,
        default="data/processed/brats_splits.csv",
        help="Output CSV path for BraTS patient splits.",
    )
    parser.add_argument(
        "--kaggle_dir",
        type=str,
        default="data/processed/kaggle_normalized",
        help="Path to normalized Kaggle directory.",
    )
    parser.add_argument(
        "--kaggle_out_csv",
        type=str,
        default="data/processed/kaggle_splits.csv",
        help="Output CSV path for Kaggle splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    brats_manifest = project_root / args.brats_manifest
    brats_out_csv = project_root / args.brats_out_csv
    kaggle_dir = project_root / args.kaggle_dir
    kaggle_out_csv = project_root / args.kaggle_out_csv

    print("=" * 60)
    print("=== BRAIN MRI DATASET SPLITTING & LEAKAGE VERIFICATION ===")
    print(f"Random Seed: {args.seed}")
    print("=" * 60)

    # 1. BraTS Patient-Level Splitting
    df_brats, leakage_pass, brats_sum = split_brats_dataset(
        manifest_path=brats_manifest,
        output_csv=brats_out_csv,
        seed=args.seed,
    )

    print("\n--- BraTS Patient-Level Dataset Splitting ---")
    print(f"Total Unique Patients: {brats_sum['unique_patients']}")
    print(f"  - Train Patients: {brats_sum['train_patients']} ({brats_sum['train_slices']} slices)")
    print(f"  - Val Patients:   {brats_sum['val_patients']} ({brats_sum['val_slices']} slices)")
    print(f"  - Test Patients:  {brats_sum['test_patients']} ({brats_sum['test_slices']} slices)")
    print(f"Saved BraTS splits manifest to: {brats_out_csv}")

    print("\n--- BraTS Data Leakage Verification Check ---")
    if leakage_pass:
        print("[RESULT]: PASS (0 patient ID overlaps across train/val/test splits)")
    else:
        print("[RESULT]: FAIL (Patient ID overlap detected between splits!)")
        sys.exit(1)


    # 2. Kaggle Dataset Splitting
    df_kaggle, kaggle_sum = split_kaggle_dataset(
        kaggle_dir=kaggle_dir,
        output_csv=kaggle_out_csv,
        seed=args.seed,
    )

    print("\n--- Kaggle Dataset Splitting ---")
    print(f"Total Kaggle Images: {kaggle_sum['total_images']}")
    print(f"  - Train Images: {kaggle_sum['train_images']} (80% of Training set)")
    print(f"  - Val Images:   {kaggle_sum['val_images']} (20% of Training set)")
    print(f"  - Test Images:  {kaggle_sum['test_images']} (100% of official Testing set)")
    print(f"Saved Kaggle splits manifest to: {kaggle_out_csv}")

    print("\n" + "=" * 60)
    print("=== FIRST FEW ROWS OF SPLIT CSVS ===")
    print("=" * 60)

    print("\n[BraTS Splits CSV (First 5 Rows)]:")
    print(df_brats[['patient_id', 'grade', 'z_index', 'has_tumor', 'split']].head())

    print("\n[Kaggle Splits CSV (First 5 Rows)]:")
    print(df_kaggle[['filename', 'class_name', 'original_folder', 'split']].head())

    print("\n=" * 60)


if __name__ == "__main__":
    main()
