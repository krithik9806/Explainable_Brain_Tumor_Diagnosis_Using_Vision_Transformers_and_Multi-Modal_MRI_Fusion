"""
End-to-End Data Preparation Pipeline Module for Brain MRI Datasets.

Integrates:
1. Normalization & Spatial Resizing (224x224)
2. Patient-Level Train / Validation / Test Dataset Splitting
3. Data Augmentation (Flips, Rotation, Elastic Deformation via Albumentations)
   - Augmentation is strictly applied ONLY to the 'train' split.
   - Validation ('val') and Test ('test') splits remain unaugmented for clean evaluation.
4. Exporting final, ready-to-train processed datasets into:
   - `data/processed/final/brats/`
   - `data/processed/final/kaggle/`
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd


def get_training_augmentation_pipeline(
    flip_prob: float = 0.5,
    rotation_degrees: int = 15,
    elastic_alpha: float = 34.0,
    elastic_sigma: float = 4.0,
    elastic_prob: float = 0.3,
) -> A.Compose:
    """
    Construct Albumentations data augmentation pipeline for training samples.

    Args:
        flip_prob (float): Probability of applying horizontal and vertical flips.
        rotation_degrees (int): Rotation angle limit (+/- degrees).
        elastic_alpha (float): Scaling factor for elastic deformation.
        elastic_sigma (float): Gaussian standard deviation for elastic deformation.
        elastic_prob (float): Probability of applying elastic deformation.

    Returns:
        A.Compose: Albumentations composition transform pipeline.
    """
    return A.Compose([
        A.HorizontalFlip(p=flip_prob),
        A.VerticalFlip(p=flip_prob),
        A.Rotate(limit=rotation_degrees, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_sigma,
            p=elastic_prob,
            border_mode=cv2.BORDER_CONSTANT,
        ),
    ])



def prepare_brats_data(
    splits_csv: Path,
    output_dir: Path,
    augment_pipeline: Optional[A.Compose] = None,
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Process BraTS dataset using split manifest CSV, applying augmentation to training samples.

    Args:
        splits_csv (Path): Path to `brats_splits.csv`.
        output_dir (Path): Destination root directory `data/processed/final/brats`.
        augment_pipeline (A.Compose): Albumentations pipeline for training set.
        max_samples (Optional[int]): Limit number of samples for quick testing runs.

    Returns:
        Dict: Execution metrics and output file records.
    """
    print(f"\n[BraTS Pipeline] Reading splits from {splits_csv} -> Saving to {output_dir}")
    if not splits_csv.exists():
        raise FileNotFoundError(f"BraTS splits file not found at {splits_csv}")

    df_splits = pd.read_csv(splits_csv)
    if max_samples is not None:
        df_splits = df_splits.head(max_samples)

    final_records = []
    total_processed = 0

    for idx, row in df_splits.iterrows():
        src_path = Path(row['file_path'])
        if not src_path.is_absolute():
            # Resolve relative to project root
            src_path = (splits_csv.parents[2] / src_path).resolve()

        if not src_path.exists():
            print(f"Warning: Source slice file {src_path} not found, skipping.")
            continue

        data = np.load(src_path)
        t1 = data['t1']
        t1ce = data['t1ce']
        t2 = data['t2']
        flair = data['flair']
        mask = data['mask']

        # Stack 4 modalities into (224, 224, 4) tensor
        img_4ch = np.stack([t1, t1ce, t2, flair], axis=-1).astype(np.float32)
        split = row['split']

        # Apply augmentation ONLY if split is 'train'
        if split == 'train' and augment_pipeline is not None:
            augmented = augment_pipeline(image=img_4ch, mask=mask)
            img_4ch = augmented['image']
            mask = augmented['mask']

        # Unstack modalities
        t1_out = img_4ch[:, :, 0]
        t1ce_out = img_4ch[:, :, 1]
        t2_out = img_4ch[:, :, 2]
        flair_out = img_4ch[:, :, 3]

        # Save to output folder structure: final/brats/<split>/<patient_id>/<slice_filename>.npz
        patient_id = row['patient_id']
        slice_name = Path(row['file_path']).name
        out_file = output_dir / split / patient_id / slice_name
        out_file.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            out_file,
            flair=flair_out,
            t1=t1_out,
            t1ce=t1ce_out,
            t2=t2_out,
            mask=mask,
        )

        record = dict(row)
        record['final_file_path'] = str(out_file.relative_to(splits_csv.parents[2]).as_posix())
        final_records.append(record)
        total_processed += 1

        if total_processed % 1000 == 0 or total_processed == len(df_splits):
            print(f"  Processed {total_processed}/{len(df_splits)} BraTS slices...")

    # Save final manifest
    df_final = pd.DataFrame(final_records)
    df_final.to_csv(output_dir / "manifest.csv", index=False)
    print(f"  Saved BraTS final manifest to: {output_dir / 'manifest.csv'}")

    return {"processed": total_processed, "records": final_records}


def prepare_kaggle_data(
    splits_csv: Path,
    output_dir: Path,
    augment_pipeline: Optional[A.Compose] = None,
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Process Kaggle dataset using split manifest CSV, applying augmentation to training samples.

    Args:
        splits_csv (Path): Path to `kaggle_splits.csv`.
        output_dir (Path): Destination root directory `data/processed/final/kaggle`.
        augment_pipeline (A.Compose): Albumentations pipeline for training set.
        max_samples (Optional[int]): Limit number of samples for quick testing runs.

    Returns:
        Dict: Execution metrics and output file records.
    """
    print(f"\n[Kaggle Pipeline] Reading splits from {splits_csv} -> Saving to {output_dir}")
    if not splits_csv.exists():
        raise FileNotFoundError(f"Kaggle splits file not found at {splits_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    df_splits = pd.read_csv(splits_csv)
    if max_samples is not None:
        df_splits = df_splits.head(max_samples)

    final_records = []
    total_processed = 0
    project_root = splits_csv.parents[2]

    for idx, row in df_splits.iterrows():
        p_str = str(row['file_path']).replace("\\", "/")
        if not p_str.startswith("data/"):
            p_str = "data/" + p_str

        src_path = (project_root / p_str).resolve()

        if not src_path.exists():
            print(f"Warning: Source Kaggle file {src_path} not found, skipping.")
            continue

        if src_path.suffix == ".npz":
            data = np.load(src_path)
            img = data['image'].astype(np.float32)
        else:
            img = cv2.imread(str(src_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        split = row['split']

        # Apply augmentation ONLY if split is 'train'
        if split == 'train' and augment_pipeline is not None:
            augmented = augment_pipeline(image=img)
            img = augmented['image']

        class_name = row['class_name']
        filename = row['filename']
        if not filename.endswith(".npz"):
            filename = Path(filename).with_suffix(".npz").name

        out_file = output_dir / split / class_name / filename
        out_file.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(out_file, image=img)

        record = dict(row)
        record['final_file_path'] = str(out_file.relative_to(project_root).as_posix())
        final_records.append(record)
        total_processed += 1

        if total_processed % 2000 == 0 or total_processed == len(df_splits):
            print(f"  Processed {total_processed}/{len(df_splits)} Kaggle images...")

    df_final = pd.DataFrame(final_records)
    df_final.to_csv(output_dir / "manifest.csv", index=False)
    print(f"  Saved Kaggle final manifest to: {output_dir / 'manifest.csv'}")

    return {"processed": total_processed, "records": final_records}



def main():
    parser = argparse.ArgumentParser(
        description="Full Data Preparation Pipeline for BraTS and Kaggle Datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--brats_splits",
        type=str,
        default="data/processed/brats_splits.csv",
        help="Path to BraTS splits CSV file.",
    )
    parser.add_argument(
        "--brats_final_out",
        type=str,
        default="data/processed/final/brats",
        help="Output directory for final BraTS dataset.",
    )
    parser.add_argument(
        "--kaggle_splits",
        type=str,
        default="data/processed/kaggle_splits.csv",
        help="Path to Kaggle splits CSV file.",
    )
    parser.add_argument(
        "--kaggle_final_out",
        type=str,
        default="data/processed/final/kaggle",
        help="Output directory for final Kaggle dataset.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of processed samples for testing.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    brats_splits = project_root / args.brats_splits
    brats_final_out = project_root / args.brats_final_out
    kaggle_splits = project_root / args.kaggle_splits
    kaggle_final_out = project_root / args.kaggle_final_out

    start_time = time.time()
    print("=" * 60)
    print("=== END-TO-END DATA PREPARATION & AUGMENTATION PIPELINE ===")
    print("=" * 60)

    aug_pipeline = get_training_augmentation_pipeline()

    # 1. Process BraTS Dataset
    brats_res = prepare_brats_data(
        splits_csv=brats_splits,
        output_dir=brats_final_out,
        augment_pipeline=aug_pipeline,
        max_samples=args.max_samples,
    )

    # 2. Process Kaggle Dataset
    kaggle_res = prepare_kaggle_data(
        splits_csv=kaggle_splits,
        output_dir=kaggle_final_out,
        augment_pipeline=aug_pipeline,
        max_samples=args.max_samples,
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("=== PREPARATION COMPLETE ===")
    print(f"BraTS Final Slices: {brats_res['processed']:,}")
    print(f"Kaggle Final Images: {kaggle_res['processed']:,}")
    print(f"Total Execution Time: {elapsed:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
