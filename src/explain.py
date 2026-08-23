"""
Visual Explainability Module using Grad-CAM for Swin Transformer MRI Classification.

This script implements Grad-CAM visual interpretability specifically adapted for Swin Transformers:
1. `swin_reshape_transform`: Converts sequence tokens [B, 49, 768] back into 2D spatial feature maps [B, 768, 7, 7].
   - Mathematical Derivation:
     Input Image = 224 x 224
     Stage 0 Patch Embedding Stem (4x4 patches) -> 56 x 56 grid (3,136 patches, 96 channels)
     Stage 1 (2x downsampling) -> 28 x 28 grid (784 patches, 192 channels)
     Stage 2 (2x downsampling) -> 14 x 14 grid (196 patches, 384 channels)
     Stage 3 & 4 (2x downsampling) -> 7 x 7 grid (49 patches, 768 channels)
     Grid Height H = sqrt(49) = 7, Grid Width W = sqrt(49) = 7
     Reshape: [B, 49, 768] -> permute/transpose -> [B, 768, 7, 7]

2. Model Loader (`load_kaggle_model`): Loads trained SwinClassifier from checkpoints/kaggle_best_model.pth.
3. Grad-CAM Generator (`generate_gradcam`): Generates activation heatmaps targeted at the final Swin stage norm layer.
4. Visualization Suite (`run_gradcam_on_kaggle_testset`): Generates and saves heatmap overlays for Kaggle test samples into results/gradcam_kaggle/.
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

from src.data.datasets import KaggleDataset
from src.fusion.fusion import pass_through_kaggle
from src.models.swin_model import SwinClassifier, build_swin_classifier
from src.utils.config_loader import load_config


def swin_reshape_transform(tensor: torch.Tensor, height: int = 7, width: int = 7) -> torch.Tensor:
    """
    Reshape transform function converting Swin Transformer sequence tokens into 2D spatial feature maps.

    Mathematical Calculation:
    -------------------------
    - Input image: [B, 3, 224, 224]
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
        # If shape is [B, H, W, C] -> permute to [B, C, H, W]
        result = tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 3:
        # Shape is [B, Num_Patches, C] -> transpose to [B, C, Num_Patches] -> reshape to [B, C, H, W]
        # Verify patch count matches height * width
        batch_size, num_patches, channels = tensor.shape
        if num_patches != height * width:
            # Fallback for unexpected token length
            grid_dim = int(np.sqrt(num_patches))
            height, width = grid_dim, grid_dim

        result = tensor.transpose(1, 2).reshape(batch_size, channels, height, width)
    else:
        raise ValueError(f"Unexpected tensor dimensionality for Swin reshape_transform: shape={tensor.shape}")

    return result


def load_kaggle_model(
    checkpoint_path: Union[str, Path] = "checkpoints/kaggle_best_model.pth",
    config_path: Union[str, Path] = "configs/kaggle_config.yaml",
    device: torch.device = torch.device("cpu"),
) -> Tuple[SwinClassifier, Dict]:
    """
    Loads trained Kaggle SwinClassifier model and config metadata from checkpoint.

    Args:
        checkpoint_path: Path to best_model.pth checkpoint.
        config_path: Path to kaggle_config.yaml configuration.
        device: Target compute device.

    Returns:
        Tuple[SwinClassifier, Dict]: Loaded model in eval mode and config dictionary.
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
    print(f"Backbone: {cfg.model.backbone} | Classes: {cfg.dataset.class_names}", flush=True)

    return model, cfg


def generate_gradcam_heatmap(
    model: SwinClassifier,
    image_tensor: torch.Tensor,
    target_category: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, int, float]:
    """
    Generates Grad-CAM activation map for a single input image tensor [3, 224, 224].

    Args:
        model (SwinClassifier): Trained SwinClassifier model.
        image_tensor (torch.Tensor): Preprocessed input tensor of shape [3, 224, 224].
        target_category (Optional[int]): Target class index for CAM computation.
        device (torch.device): Compute device.

    Returns:
        Tuple[np.ndarray, int, float]: (grayscale_cam [224, 224], pred_class_idx, confidence)
    """
    model.eval()

    # Target final norm layer of last Swin stage
    timm_backbone = model.backbone.backbone
    if hasattr(timm_backbone, "layers") and len(timm_backbone.layers) > 0:
        target_layers = [timm_backbone.layers[-1].blocks[-1].norm2]
    elif hasattr(timm_backbone, "norm"):
        target_layers = [timm_backbone.norm]
    else:
        raise AttributeError("Unable to locate final normalization layer in Swin Transformer backbone.")

    # Prepare input batch [1, 3, 224, 224]
    input_batch = image_tensor.unsqueeze(0).to(device)

    # Compute prediction & softmax confidence
    with torch.no_grad():
        logits = model(input_batch)
        probs = torch.softmax(logits, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class_idx].item()

    if target_category is None:
        target_category = pred_class_idx

    targets = [ClassifierOutputTarget(target_category)]

    # Instantiate GradCAM with Swin reshape_transform
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=swin_reshape_transform,
    )

    # Generate CAM mask [1, 224, 224]
    grayscale_cam = cam(input_tensor=input_batch, targets=targets)[0, :]
    return grayscale_cam, pred_class_idx, confidence


def create_heatmap_overlay(
    rgb_image: np.ndarray,
    grayscale_cam: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blends grayscale CAM heatmap over original RGB image using OpenCV JET colormap.

    Args:
        rgb_image (np.ndarray): Normalized RGB image array of shape [224, 224, 3] in range [0, 1].
        grayscale_cam (np.ndarray): Grad-CAM heatmap array of shape [224, 224] in range [0, 1].
        alpha (float): Overlay blend transparency weight (0.0 to 1.0).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (colored_heatmap_rgb, blended_overlay_rgb)
    """
    # Ensure inputs are float32 in [0, 1]
    rgb_img = np.clip(rgb_image, 0.0, 1.0)
    cam = np.clip(grayscale_cam, 0.0, 1.0)

    # Convert CAM to 8-bit integer and apply JET colormap
    cam_uint8 = np.uint8(255 * cam)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Blend original image with heatmap
    blended_rgb = alpha * heatmap_rgb + (1.0 - alpha) * rgb_img
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

    # 1. Load trained model & config
    model, cfg = load_kaggle_model(checkpoint_path=checkpoint_path, config_path=config_path, device=device)
    class_names = cfg.dataset.class_names

    # 2. Output directory setup
    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Read Kaggle test split CSV
    csv_file = PROJECT_ROOT / splits_csv
    if not csv_file.exists():
        raise FileNotFoundError(f"Splits file not found at: {csv_file}")

    df_splits = pd.read_csv(csv_file)
    df_test = df_splits[df_splits["split"].str.lower() == "test"].reset_index(drop=True)
    print(f"Found {len(df_test)} test samples in {splits_csv}", flush=True)

    # Sample images per class
    sampled_records = []
    for cls in class_names:
        sub_df = df_test[df_test["class_name"] == cls]
        if len(sub_df) > 0:
            sampled_records.extend(sub_df.head(num_samples_per_class).to_dict("records"))

    print(f"Selected {len(sampled_records)} test samples for Grad-CAM visualization.", flush=True)

    summary_results = []

    # 4. Generate Grad-CAM heatmaps for each sample
    for idx, rec in enumerate(sampled_records, start=1):
        rel_path = rec["file_path"]
        true_class = rec["class_name"]
        npz_file = (PROJECT_ROOT / "data" / rel_path).resolve()

        if not npz_file.exists():
            npz_file = (PROJECT_ROOT / rel_path).resolve()

        if not npz_file.exists():
            print(f"Warning: File {rel_path} not found, skipping.")
            continue

        # Load raw image from .npz
        npz_data = np.load(npz_file)
        img_key = "image" if "image" in npz_data else list(npz_data.keys())[0]
        raw_img = npz_data[img_key]

        # Pass-through tensor [3, 224, 224]
        img_tensor = pass_through_kaggle(raw_img, return_tensor=True)

        # Generate Grad-CAM
        grayscale_cam, pred_idx, conf = generate_gradcam_heatmap(
            model=model,
            image_tensor=img_tensor,
            target_category=None,  # Defaults to predicted class
            device=device,
        )

        pred_class = class_names[pred_idx]
        is_correct = pred_class == true_class

        # Unnormalize ImageNet tensor [3, 224, 224] to [0, 1] RGB image for clean display
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_img = img_tensor.numpy().transpose(1, 2, 0) * std + mean
        rgb_img = np.clip(rgb_img, 0.0, 1.0)

        heatmap_rgb, blended_rgb = create_heatmap_overlay(rgb_img, grayscale_cam, alpha=0.5)

        # Plot comparison figure: [Original MRI | Grad-CAM Heatmap | Blended Overlay]
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

    print("\n" + "=" * 80)
    print("=== GRAD-CAM EXPLAINABILITY PIPELINE COMPLETE ===")
    print(f"Generated {len(summary_results)} visual explainability overlays in: {out_dir.resolve()}")
    print("=" * 80)

    return summary_results


def main():
    parser = argparse.ArgumentParser(description="Run Grad-CAM Visual Explainability for Kaggle Swin Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/kaggle_best_model.pth",
        help="Path to Kaggle trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kaggle_config.yaml",
        help="Path to Kaggle config file",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="data/processed/kaggle_splits.csv",
        help="Path to Kaggle splits CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/gradcam_kaggle",
        help="Output directory for Grad-CAM plots",
    )

    args = parser.parse_args()
    run_gradcam_on_kaggle_testset(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        splits_csv=args.splits,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
