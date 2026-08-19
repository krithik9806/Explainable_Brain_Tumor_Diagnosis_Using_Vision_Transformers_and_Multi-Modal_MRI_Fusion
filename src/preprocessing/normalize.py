"""
Intensity Normalization and Spatial Resizing Module for Brain MRI Datasets.

This script processes 2D MRI slices from both BraTS and Kaggle datasets:
1. BraTS: Applies per-modality Z-score normalization ((x - mean) / std) over non-zero brain regions
   and resizes modalities to 224x224 (bilinear for modalities, nearest-neighbor for segmentation masks).
2. Kaggle: Applies ImageNet-style standard normalization ((x - mean) / std) / [0, 1] scaling
   and resizes single-modality images to 224x224 (bilinear).
3. Saves output datasets into `data/processed/brats_normalized/` and `data/processed/kaggle_normalized/`.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


# ImageNet statistics for 3-channel vision backbones (Swin Transformer)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_image(img: np.ndarray, target_size: Tuple[int, int] = (224, 224), is_mask: bool = False) -> np.ndarray:
    """
    Resize a 2D image or segmentation mask to the target size.

    Args:
        img (np.ndarray): Input 2D or 3D numpy array.
        target_size (Tuple[int, int]): Desired (width, height) output size. Default is (224, 224).
        is_mask (bool): If True, uses nearest-neighbor interpolation to preserve discrete label values.
                        If False, uses bilinear interpolation for smooth image intensity scaling.

    Returns:
        np.ndarray: Resized numpy array.
    """
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(img, target_size, interpolation=interp)
    return resized


def z_score_normalize_brats(img: np.ndarray, non_zero_only: bool = True, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score intensity normalization for a BraTS MRI modality slice.

    Computes zero-mean, unit-variance normalization: (x - mean) / (std + eps).
    When non_zero_only is True, statistics are computed strictly over the non-zero (brain) region,
    leaving background voxels set to 0.

    Args:
        img (np.ndarray): Input 2D float intensity slice.
        non_zero_only (bool): If True, compute mean and std only over non-zero brain voxels.
        eps (float): Small epsilon constant to prevent division by zero.

    Returns:
        np.ndarray: Z-score normalized float32 slice array.
    """
    img = img.astype(np.float32)
    if non_zero_only and np.any(img > 0):
        brain_mask = img > 0
        mean_val = img[brain_mask].mean()
        std_val = img[brain_mask].std()
        normalized = np.zeros_like(img, dtype=np.float32)
        normalized[brain_mask] = (img[brain_mask] - mean_val) / (std_val + eps)
        return normalized
    else:
        mean_val = img.mean()
        std_val = img.std()
        if std_val < eps:
            return np.zeros_like(img, dtype=np.float32)
        return ((img - mean_val) / (std_val + eps)).astype(np.float32)


def normalize_kaggle_image(img: np.ndarray, method: str = "imagenet") -> np.ndarray:
    """
    Standard intensity normalization for Kaggle 2D brain MRI images.

    Args:
        img (np.ndarray): Input image array (H, W) or (H, W, C), uint8 or float.
        method (str): 'imagenet' for ImageNet z-score normalization, or 'minmax' for [0, 1] scaling.

    Returns:
        np.ndarray: Normalized float32 numpy array.
    """
    # Convert uint8 to float32 [0.0, 1.0]
    img_float = img.astype(np.float32)
    if img_float.max() > 1.0:
        img_float /= 255.0

    if method.lower() == "minmax":
        return img_float

    elif method.lower() == "imagenet":
        # Ensure 3-channel image for Swin Transformer compatibility
        if img_float.ndim == 2:
            img_float = np.stack([img_float] * 3, axis=-1)
        elif img_float.ndim == 3 and img_float.shape[2] == 1:
            img_float = np.concatenate([img_float] * 3, axis=-1)

        normalized = (img_float - IMAGENET_MEAN) / IMAGENET_STD
        return normalized.astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization method '{method}'. Choose 'imagenet' or 'minmax'.")


def get_array_stats(arr: np.ndarray) -> Dict[str, float]:
    """Calculate min, max, mean, std statistics for an array."""
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "shape": list(arr.shape),
    }


def process_brats_dataset(
    input_dir: Path, output_dir: Path, target_size: Tuple[int, int] = (224, 224)
) -> Dict:
    """Process all BraTS slice .npz files: resize to 224x224 and Z-score normalize modalities."""
    print(f"\n[BraTS Processing] Reading from {input_dir} -> Saving to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    stats_samples = []
    total_slices = 0

    for idx, p_dir in enumerate(patient_dirs, start=1):
        out_p_dir = output_dir / p_dir.name
        out_p_dir.mkdir(parents=True, exist_ok=True)

        slice_files = sorted(list(p_dir.glob("*.npz")))
        for s_file in slice_files:
            data = np.load(s_file)
            
            # Load raw modalities & mask
            modalities = {m: data[m] for m in ['t1', 't1ce', 't2', 'flair']}
            mask = data['mask']

            # Capture stats for sample (first patient, middle slice)
            if idx == 1 and total_slices == len(slice_files) // 2:
                for m_name, m_arr in modalities.items():
                    stats_samples.append({
                        "dataset": "BraTS",
                        "sample": f"{p_dir.name}/{s_file.name}",
                        "item": m_name,
                        "before": get_array_stats(m_arr),
                    })

            # Resize & Normalize
            norm_modalities = {}
            for m_name, m_arr in modalities.items():
                resized = resize_image(m_arr, target_size=target_size, is_mask=False)
                normalized = z_score_normalize_brats(resized, non_zero_only=True)
                norm_modalities[m_name] = normalized

            resized_mask = resize_image(mask, target_size=target_size, is_mask=True)

            # Record after stats for sample
            if idx == 1 and total_slices == len(slice_files) // 2:
                for s_entry in stats_samples:
                    if s_entry["dataset"] == "BraTS":
                        m_name = s_entry["item"]
                        s_entry["after"] = get_array_stats(norm_modalities[m_name])

            # Save normalized output .npz
            out_s_file = out_p_dir / s_file.name
            np.savez_compressed(
                out_s_file,
                flair=norm_modalities['flair'],
                t1=norm_modalities['t1'],
                t1ce=norm_modalities['t1ce'],
                t2=norm_modalities['t2'],
                mask=resized_mask,
            )
            total_slices += 1

        if idx % 30 == 0 or idx == len(patient_dirs):
            print(f"  Processed {idx}/{len(patient_dirs)} BraTS patient directories ({total_slices} total slices)")

    # Copy manifest if exists
    manifest_src = input_dir / "manifest.csv"
    if manifest_src.exists():
        manifest_df = pd.read_csv(manifest_src)
        manifest_df.to_csv(output_dir / "manifest.csv", index=False)
        print(f"  Copied manifest.csv to {output_dir / 'manifest.csv'}")

    return {"processed_slices": total_slices, "stats": stats_samples}


def process_kaggle_dataset(
    input_dir: Path, output_dir: Path, target_size: Tuple[int, int] = (224, 224)
) -> Dict:
    """Process all Kaggle slice images: resize to 224x224 and ImageNet normalize."""
    print(f"\n[Kaggle Processing] Reading from {input_dir} -> Saving to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old uncompressed .npy files if present
    old_npy_files = list(output_dir.glob("**/*.npy"))
    if old_npy_files:
        print(f"  Removing {len(old_npy_files)} old uncompressed .npy files from {output_dir}...")
        for old_file in old_npy_files:
            try:
                old_file.unlink()
            except Exception as e:
                print(f"  Warning: failed to delete {old_file}: {e}")

    image_files = sorted(list(input_dir.glob("**/*.jpg")) + list(input_dir.glob("**/*.png")))
    stats_samples = []
    total_images = 0

    for idx, img_path in enumerate(image_files, start=1):
        rel_path = img_path.relative_to(input_dir)
        out_file = (output_dir / rel_path).with_suffix(".npz")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if idx in [1, 100, 1000]:
            sample_before = get_array_stats(img_rgb)
        else:
            sample_before = None

        resized = resize_image(img_rgb, target_size=target_size, is_mask=False)
        normalized = normalize_kaggle_image(resized, method="imagenet")

        if sample_before is not None:
            stats_samples.append({
                "dataset": "Kaggle",
                "sample": str(rel_path),
                "item": "image_rgb",
                "before": sample_before,
                "after": get_array_stats(normalized),
            })

        # Save float32 normalized array as compressed .npz with key 'image'
        np.savez_compressed(out_file, image=normalized)
        total_images += 1

        if idx % 2000 == 0 or idx == len(image_files):
            print(f"  Processed {idx}/{len(image_files)} Kaggle images")

    return {"processed_images": total_images, "stats": stats_samples}


def main():
    parser = argparse.ArgumentParser(
        description="Normalize and resize BraTS and Kaggle brain MRI datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--brats_in",
        type=str,
        default="data/processed/brats_slices",
        help="Input directory for extracted BraTS 2D slices.",
    )
    parser.add_argument(
        "--brats_out",
        type=str,
        default="data/processed/brats_normalized",
        help="Output directory for normalized BraTS slices.",
    )
    parser.add_argument(
        "--kaggle_in",
        type=str,
        default="data/raw/kaggle",
        help="Input directory for raw Kaggle dataset.",
    )
    parser.add_argument(
        "--kaggle_out",
        type=str,
        default="data/processed/kaggle_normalized",
        help="Output directory for normalized Kaggle images.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target (width, height) spatial dimension.",
    )

    args = parser.parse_args()
    target_size = (args.target_size[0], args.target_size[1])

    project_root = Path(__file__).resolve().parents[2]
    brats_in = project_root / args.brats_in
    brats_out = project_root / args.brats_out
    kaggle_in = project_root / args.kaggle_in
    kaggle_out = project_root / args.kaggle_out

    start_time = time.time()
    print("=" * 60)
    print("=== DATASET NORMALIZATION AND RESIZING PIPELINE ===")
    print(f"Target Resolution: {target_size}")
    print("=" * 60)

    # 1. Process BraTS
    brats_res = process_brats_dataset(brats_in, brats_out, target_size=target_size)

    # 2. Process Kaggle
    kaggle_res = process_kaggle_dataset(kaggle_in, kaggle_out, target_size=target_size)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("=== NORMALIZATION COMPLETE ===")
    print(f"BraTS Processed Slices: {brats_res['processed_slices']:,}")
    print(f"Kaggle Processed Images: {kaggle_res['processed_images']:,}")
    print(f"Total Execution Time: {elapsed:.2f} seconds")
    print("=" * 60)

    # Print Intensity Statistics
    print("\n" + "=" * 60)
    print("=== BEFORE / AFTER INTENSITY STATISTICS SUMMARY ===")
    print("=" * 60)

    all_stats = brats_res["stats"] + kaggle_res["stats"]
    for s in all_stats:
        print(f"\nDataset: {s['dataset']} | Sample: {s['sample']} | Item: {s['item']}")
        b, a = s["before"], s["after"]
        print(f"  BEFORE -> Shape: {b['shape']} | Min: {b['min']:.4f} | Max: {b['max']:.4f} | Mean: {b['mean']:.4f} | Std: {b['std']:.4f}")
        print(f"  AFTER  -> Shape: {a['shape']} | Min: {a['min']:.4f} | Max: {a['max']:.4f} | Mean: {a['mean']:.4f} | Std: {a['std']:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
