"""
Visual Explainability Module using Grad-CAM for Swin Transformer MRI Classification.

This script implements Grad-CAM visual interpretability adapted for Swin Transformers across both:
1. Kaggle Single-Modality 3-Channel Classifier (`run_gradcam_on_kaggle_testset`)
2. BraTS Multi-Modal 4-Channel Fusion Classifier (`run_gradcam_on_brats_testset`)

Technical Architecture & Math:
------------------------------
`swin_reshape_transform`: Converts sequence tokens [B, 49, 768] back into 2D spatial feature maps [B, 768, 7, 7].
- Input Image = 224 x 224
- Stage 0 Patch Embedding Stem (4x4 patches) -> 56 x 56 grid (3,136 patches, 96 channels)
- Stage 1 (2x downsampling) -> 28 x 28 grid (784 patches, 192 channels)
- Stage 2 (2x downsampling) -> 14 x 14 grid (196 patches, 384 channels)
- Stage 3 & 4 (2x downsampling) -> 7 x 7 grid (49 patches, 768 channels)
- Grid Height H = sqrt(49) = 7, Grid Width W = sqrt(49) = 7
- Reshape: [B, 49, 768] -> transpose/reshape -> [B, 768, 7, 7]

Representative Modality Selection Rationale (BraTS):
---------------------------------------------------
FLAIR (Fluid-Attenuated Inversion Recovery) is explicitly chosen as the display background image for 4-channel fusion scans.
In clinical neuro-radiology, FLAIR suppresses hyperintense signal from normal cerebrospinal fluid (CSF) while accentuating
peritumoral edema and tissue lesions with high contrast, making it the gold-standard modality for visual tumor inspection.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import BraTSDataset, KaggleDataset
from src.fusion.fusion import fuse_brats_modalities, pass_through_kaggle
from src.models.swin_model import SwinClassifier, build_swin_classifier
from src.utils.config_loader import load_config


def swin_reshape_transform(tensor: torch.Tensor, height: int = 7, width: int = 7) -> torch.Tensor:
    """
    Reshape transform function converting Swin Transformer sequence tokens into 2D spatial feature maps.

    Mathematical Calculation:
    -------------------------
    - Input image: [B, C, 224, 224] (C=3 or C=4)
    - Patch embedding (4x4 patch): 224 / 4 = 56 x 56 grid
    - Stage 1 patch merging (2x downsampling): 56 / 2 = 28 x 28 grid
    - Stage 2 patch merging (2x downsampling): 28 / 2 = 14 x 14 grid
    - Stage 3 patch merging (2x downsampling): 14 / 2 = 7 x 7 grid
    - Output token sequence from stage 4 target layer: [B, 49, 768] (49 tokens = 7 x 7)

    Args:
        tensor (torch.Tensor): Token sequence tensor of shape [B, 49, 768] or [B, 7, 7, 768].
        height (int): Target spatial grid height (default: 7).
        width (int): Target spatial grid width (default: 7).

    Returns:
        torch.Tensor: 4D spatial feature map of shape [B, 768, 7, 7].
    """
    if tensor.ndim == 4:
        result = tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 3:
        batch_size, num_patches, channels = tensor.shape
        if num_patches != height * width:
            grid_dim = int(np.sqrt(num_patches))
            height, width = grid_dim, grid_dim

        result = tensor.transpose(1, 2).reshape(batch_size, channels, height, width)
    else:
        raise ValueError(f"Unexpected tensor dimensionality for Swin reshape_transform: shape={tensor.shape}")

    return result


def load_model_from_config(
    checkpoint_path: Union[str, Path],
    config_path: Union[str, Path],
    device: torch.device = torch.device("cpu"),
) -> Tuple[SwinClassifier, Dict]:
    """
    Loads trained SwinClassifier model and config metadata from checkpoint.
    """
    cfg = load_config(config_path)
    ckpt_file = PROJECT_ROOT / checkpoint_path if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_file}")

    model = build_swin_classifier(
        backbone_name=cfg.model.backbone,
        input_channels=cfg.dataset.input_channels,
        num_classes=cfg.dataset.num_classes,
        pretrained=False,
    )

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"=== Successfully Loaded Model from {ckpt_file.name} ===", flush=True)
    print(f"Backbone: {cfg.model.backbone} | Channels: {cfg.dataset.input_channels} | Classes: {cfg.dataset.class_names}", flush=True)

    return model, cfg


def generate_gradcam_heatmap(
    model: SwinClassifier,
    image_tensor: torch.Tensor,
    target_category: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, int, float]:
    """
    Generates Grad-CAM activation map for a single input image tensor [C, 224, 224].
    """
    model.eval()

    timm_backbone = model.backbone.backbone
    if hasattr(timm_backbone, "layers") and len(timm_backbone.layers) > 0:
        target_layers = [timm_backbone.layers[-1].blocks[-1].norm2]
    elif hasattr(timm_backbone, "norm"):
        target_layers = [timm_backbone.norm]
    else:
        raise AttributeError("Unable to locate final normalization layer in Swin Transformer backbone.")

    input_batch = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_batch)
        probs = torch.softmax(logits, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class_idx].item()

    if target_category is None:
        target_category = pred_class_idx

    targets = [ClassifierOutputTarget(target_category)]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=swin_reshape_transform,
    )

    grayscale_cam = cam(input_tensor=input_batch, targets=targets)[0, :]
    return grayscale_cam, pred_class_idx, confidence


def create_heatmap_overlay(
    gray_image: np.ndarray,
    grayscale_cam: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blends grayscale CAM heatmap over single-channel or RGB image array using OpenCV JET colormap.
    """
    if gray_image.ndim == 2:
        # Convert single-channel to RGB
        base_img = np.stack([gray_image] * 3, axis=-1)
    else:
        base_img = gray_image

    base_img = np.clip(base_img, 0.0, 1.0)
    cam = np.clip(grayscale_cam, 0.0, 1.0)

    cam_uint8 = np.uint8(255 * cam)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    blended_rgb = alpha * heatmap_rgb + (1.0 - alpha) * base_img
    blended_rgb = np.clip(blended_rgb, 0.0, 1.0)

    return heatmap_rgb, blended_rgb


def run_gradcam_on_kaggle_testset(
    checkpoint_path: str = "checkpoints/kaggle_best_model.pth",
    config_path: str = "configs/kaggle_config.yaml",
    splits_csv: str = "data/processed/kaggle_splits.csv",
    output_dir: str = "results/gradcam_kaggle",
    num_samples_per_class: int = 2,
):
    """
    Executes Grad-CAM visual explainability pipeline on Kaggle test set samples across all 4 classes.
    """
    print("=" * 80)
    print("=== DAY 23: GRAD-CAM VISUAL EXPLAINABILITY (KAGGLE SWIN TRANSFORMER) ===")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}", flush=True)

    model, cfg = load_model_from_config(checkpoint_path=checkpoint_path, config_path=config_path, device=device)
    class_names = cfg.dataset.class_names

    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_file = PROJECT_ROOT / splits_csv
    if not csv_file.exists():
        raise FileNotFoundError(f"Splits file not found at: {csv_file}")

    df_splits = pd.read_csv(csv_file)
    df_test = df_splits[df_splits["split"].str.lower() == "test"].reset_index(drop=True)

    sampled_records = []
    for cls in class_names:
        sub_df = df_test[df_test["class_name"] == cls]
        if len(sub_df) > 0:
            sampled_records.extend(sub_df.head(num_samples_per_class).to_dict("records"))

    print(f"Selected {len(sampled_records)} Kaggle test samples for Grad-CAM visualization.", flush=True)

    summary_results = []

    for idx, rec in enumerate(sampled_records, start=1):
        rel_path = rec["file_path"]
        true_class = rec["class_name"]
        npz_file = (PROJECT_ROOT / "data" / rel_path).resolve()
        if not npz_file.exists():
            npz_file = (PROJECT_ROOT / rel_path).resolve()

        if not npz_file.exists():
            continue

        npz_data = np.load(npz_file)
        img_key = "image" if "image" in npz_data else list(npz_data.keys())[0]
        raw_img = npz_data[img_key]

        img_tensor = pass_through_kaggle(raw_img, return_tensor=True)

        grayscale_cam, pred_idx, conf = generate_gradcam_heatmap(
            model=model,
            image_tensor=img_tensor,
            device=device,
        )

        pred_class = class_names[pred_idx]
        is_correct = pred_class == true_class

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_img = img_tensor.numpy().transpose(1, 2, 0) * std + mean
        rgb_img = np.clip(rgb_img, 0.0, 1.0)

        heatmap_rgb, blended_rgb = create_heatmap_overlay(rgb_img, grayscale_cam, alpha=0.5)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Original MRI\nTrue: {true_class}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(grayscale_cam, cmap="jet")
        axes[1].set_title(f"Grad-CAM Heatmap\n(Target: {pred_class})", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        color_str = "green" if is_correct else "red"
        axes[2].imshow(blended_rgb)
        axes[2].set_title(
            f"Blended Overlay\nPred: {pred_class} ({conf * 100:.1f}%)",
            fontsize=11,
            fontweight="bold",
            color=color_str,
        )
        axes[2].axis("off")

        plt.suptitle(
            f"Swin-Tiny Grad-CAM Explanation | Sample #{idx} [{true_class}]",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()

        sample_name = Path(rel_path).stem
        out_path = out_dir / f"gradcam_{idx:02d}_{true_class}_{sample_name}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        status_str = "CORRECT" if is_correct else "MISCLASSIFIED"
        print(
            f"[{idx}/{len(sampled_records)}] True: {true_class:<10} | Pred: {pred_class:<10} "
            f"({conf * 100:.1f}%) -> {status_str:<12} | Saved: {out_path.name}",
            flush=True,
        )

        summary_results.append({
            "sample_id": idx,
            "true_class": true_class,
            "pred_class": pred_class,
            "confidence": conf,
            "is_correct": is_correct,
            "output_path": str(out_path.relative_to(PROJECT_ROOT).as_posix()),
        })

    print(f"\n=== Kaggle Grad-CAM Completed: {len(summary_results)} figures saved in {out_dir.resolve()} ===\n")
    return summary_results


def run_gradcam_on_brats_testset(
    checkpoint_path: str = "checkpoints/brats_best_model.pth",
    config_path: str = "configs/brats_fusion_config.yaml",
    splits_csv: str = "data/processed/brats_splits.csv",
    output_dir: str = "results/gradcam_brats",
    num_samples_per_grade: int = 3,
):
    """
    Executes Grad-CAM visual explainability pipeline on BraTS 4-channel MRI fusion model,
    overlaying heatmaps onto representative FLAIR slices and comparing against ground-truth tumor masks.
    """
    print("=" * 80)
    print("=== DAY 24: GRAD-CAM VISUAL EXPLAINABILITY (BRATS 4-CHANNEL FUSION MODEL) ===")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}", flush=True)

    # 1. Load trained model & config
    model, cfg = load_model_from_config(checkpoint_path=checkpoint_path, config_path=config_path, device=device)
    class_names = cfg.dataset.class_names  # ["LGG", "HGG"]

    # 2. Output directory setup
    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Read BraTS test split CSV
    csv_file = PROJECT_ROOT / splits_csv
    if not csv_file.exists():
        raise FileNotFoundError(f"Splits file not found at: {csv_file}")

    df_splits = pd.read_csv(csv_file)
    df_test = df_splits[df_splits["split"].str.lower() == "test"].reset_index(drop=True)
    print(f"Found {len(df_test)} BraTS test slices in {splits_csv}", flush=True)

    # Filter slices that contain actual non-zero ground-truth tumor masks for meaningful evaluation
    sampled_records = []
    for grade in class_names:
        sub_df = df_test[df_test["grade"] == grade]

        # Prioritize slices with positive tumor area
        selected = []
        for _, row in sub_df.iterrows():
            fp = (PROJECT_ROOT / "data" / row["file_path"]).resolve()
            if not fp.exists():
                fp = (PROJECT_ROOT / row["file_path"]).resolve()

            if fp.exists():
                data = np.load(fp)
                if np.sum(data["mask"]) > 50:  # Slice has substantial tumor tissue
                    selected.append(row.to_dict())
                if len(selected) >= num_samples_per_grade:
                    break

        # Fallback if fewer tumor slices found
        if len(selected) < num_samples_per_grade:
            for _, row in sub_df.iterrows():
                r_dict = row.to_dict()
                if r_dict not in selected:
                    selected.append(r_dict)
                if len(selected) >= num_samples_per_grade:
                    break

        sampled_records.extend(selected)

    print(f"Selected {len(sampled_records)} BraTS test slices (HGG & LGG) with ground-truth masks.", flush=True)

    summary_results = []

    # 4. Generate 4-panel comparison plots for each BraTS slice
    for idx, rec in enumerate(sampled_records, start=1):
        rel_path = rec["file_path"]
        true_grade = rec["grade"]
        patient_id = rec.get("patient_id", "patient")

        npz_file = (PROJECT_ROOT / "data" / rel_path).resolve()
        if not npz_file.exists():
            npz_file = (PROJECT_ROOT / rel_path).resolve()

        if not npz_file.exists():
            print(f"Warning: Slice file {rel_path} not found, skipping.")
            continue

        npz_data = np.load(npz_file)
        t1 = npz_data["t1"]
        t1ce = npz_data["t1ce"]
        t2 = npz_data["t2"]
        flair = npz_data["flair"]
        mask = npz_data["mask"]

        # Apply Day 11 Early Fusion to produce 4-channel tensor [4, 224, 224]
        fused_tensor = fuse_brats_modalities(t1, t1ce, t2, flair, return_tensor=True)

        # Generate Grad-CAM activation heatmap
        grayscale_cam, pred_idx, conf = generate_gradcam_heatmap(
            model=model,
            image_tensor=fused_tensor,
            device=device,
        )

        pred_grade = class_names[pred_idx]
        is_correct = pred_grade == true_grade

        # RATIONALE: Normalize representative FLAIR slice [224, 224] to [0, 1] for visualization
        flair_min, flair_max = flair.min(), flair.max()
        flair_norm = (flair - flair_min) / max((flair_max - flair_min), 1e-6)

        # Blend FLAIR slice with Grad-CAM heatmap
        heatmap_rgb, blended_rgb = create_heatmap_overlay(flair_norm, grayscale_cam, alpha=0.5)

        # 4-Panel Figure Plotting:
        # Panel 1: Representative FLAIR MRI Slice
        # Panel 2: Ground-Truth Tumor Mask (Expert Annotation)
        # Panel 3: Grad-CAM Heatmap (Model Attention)
        # Panel 4: Grad-CAM Heatmap + Ground-Truth Tumor Contour Overlay
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

        # Panel 1: Representative FLAIR MRI
        axes[0].imshow(flair_norm, cmap="gray")
        axes[0].set_title(f"FLAIR MRI (Display Modality)\nTrue Grade: {true_grade}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Ground-Truth Tumor Mask
        axes[1].imshow(mask, cmap="Reds")
        axes[1].set_title("Ground-Truth Tumor Mask\n(Expert Radiologist Annotation)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Grad-CAM Heatmap
        axes[2].imshow(grayscale_cam, cmap="jet")
        axes[2].set_title(f"Grad-CAM Heatmap\n(Target Grade: {pred_grade})", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        # Panel 4: Heatmap + Ground-Truth Contour Overlay
        color_str = "green" if is_correct else "red"
        axes[3].imshow(blended_rgb)
        if np.sum(mask) > 0:
            axes[3].contour(mask, levels=[0.5], colors=["magenta"], linewidths=2)
        axes[3].set_title(
            f"Heatmap + Tumor Contour\nPred: {pred_grade} ({conf * 100:.1f}%)",
            fontsize=11,
            fontweight="bold",
            color=color_str,
        )
        axes[3].axis("off")

        plt.suptitle(
            f"BraTS 4-Channel Swin Grad-CAM vs Tumor Mask | {patient_id} [{true_grade}]",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()

        slice_name = Path(rel_path).stem
        out_path = out_dir / f"gradcam_brats_{idx:02d}_{true_grade}_{patient_id}_{slice_name}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        status_str = "CORRECT" if is_correct else "MISCLASSIFIED"
        print(
            f"[{idx}/{len(sampled_records)}] Patient: {patient_id:<12} | True: {true_grade:<4} | "
            f"Pred: {pred_grade:<4} ({conf * 100:.1f}%) -> {status_str:<12} | Saved: {out_path.name}",
            flush=True,
        )

        summary_results.append({
            "sample_id": idx,
            "patient_id": patient_id,
            "true_grade": true_grade,
            "pred_grade": pred_grade,
            "confidence": conf,
            "is_correct": is_correct,
            "output_path": str(out_path.relative_to(PROJECT_ROOT).as_posix()),
        })

    print("\n" + "=" * 80)
    print("=== BRATS GRAD-CAM EXPLAINABILITY PIPELINE COMPLETE ===")
    print(f"Generated {len(summary_results)} 4-panel visual comparison figures in: {out_dir.resolve()}")
    print("=" * 80)

    return summary_results


def main():
    parser = argparse.ArgumentParser(description="Run Grad-CAM Visual Explainability for Swin Transformer Models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["kaggle", "brats", "all"],
        help="Target dataset to explain: 'kaggle', 'brats', or 'all'",
    )
    parser.add_argument(
        "--kaggle_checkpoint",
        type=str,
        default="checkpoints/kaggle_best_model.pth",
        help="Path to Kaggle model checkpoint",
    )
    parser.add_argument(
        "--brats_checkpoint",
        type=str,
        default="checkpoints/brats_best_model.pth",
        help="Path to BraTS model checkpoint",
    )

    args = parser.parse_args()

    if args.dataset in ["kaggle", "all"]:
        run_gradcam_on_kaggle_testset(checkpoint_path=args.kaggle_checkpoint)

    if args.dataset in ["brats", "all"]:
        run_gradcam_on_brats_testset(checkpoint_path=args.brats_checkpoint)


if __name__ == "__main__":
    main()
