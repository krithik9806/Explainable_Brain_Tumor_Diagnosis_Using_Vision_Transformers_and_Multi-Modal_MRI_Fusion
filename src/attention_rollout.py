"""
Attention Rollout Module & Explainability Comparison Suite for Swin Transformer Models.

This module implements Attention Rollout (Abnar & Zuidema, 2020) specifically adapted for Swin Transformers:
1. Shifted-Window Attention Handling:
   Swin Transformer uses local windowed self-attention (7x7 = 49 tokens per window) across 4 spatial hierarchy stages:
   - Stage 0 (56x56 grid): 64 windows of 7x7 tokens
   - Stage 1 (28x28 grid): 16 windows of 7x7 tokens
   - Stage 2 (14x14 grid): 4 windows of 7x7 tokens
   - Stage 3 (7x7 grid): 1 window of 7x7 tokens

2. Rollout Algorithm:
   - Registers forward hooks on `b.attn.softmax` across all 12 Swin Transformer blocks.
   - For each block, averages attention weights across heads, adds identity matrix for residual connections (0.5 * A + 0.5 * I),
     normalizes rows, and extracts token self-importance.
   - Reconstructs 2D spatial grids for each stage, interpolates to [224, 224], and accumulates across all 12 blocks.

3. Side-by-Side Explainability Comparison:
   - Generates 3-panel visual figures: [Original MRI / FLAIR | Grad-CAM Heatmap | Attention Rollout Heatmap]
   - Saves output comparison plots into results/explainability_comparison/ for Kaggle and BraTS test samples.
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
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import BraTSDataset, KaggleDataset
from src.explain import (
    create_heatmap_overlay,
    generate_gradcam_heatmap,
    load_model_from_config,
)
from src.fusion.fusion import fuse_brats_modalities, pass_through_kaggle
from src.models.swin_model import SwinClassifier


class SwinAttentionRollout:
    """
    Computes Attention Rollout saliency maps for Swin Transformer architectures.
    """

    def __init__(self, model: SwinClassifier, device: torch.device = torch.device("cpu")):
        self.model = model.to(device)
        self.device = device
        self.attn_maps = []
        self.hooks = []

    def _register_hooks(self):
        """
        Registers forward hooks on `block.attn.softmax` across all Swin Transformer blocks.
        """
        self.attn_maps = []
        self.hooks = []

        timm_backbone = self.model.backbone.backbone
        if hasattr(timm_backbone, "layers"):
            for stage in timm_backbone.layers:
                for block in stage.blocks:
                    if hasattr(block.attn, "softmax"):
                        hook = block.attn.softmax.register_forward_hook(self._hook_fn)
                        self.hooks.append(hook)

    def _hook_fn(self, module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
        self.attn_maps.append(output.detach())

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_rollout(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Executes forward pass with hooks and computes accumulated Attention Rollout saliency map [224, 224].

        Args:
            image_tensor (torch.Tensor): Input image tensor of shape [C, 224, 224].

        Returns:
            np.ndarray: Normalized Attention Rollout heatmap array of shape [224, 224] in range [0, 1].
        """
        self.model.eval()
        self._register_hooks()

        input_batch = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            _ = self.model(input_batch)

        self._remove_hooks()

        if len(self.attn_maps) == 0:
            raise RuntimeError("No attention maps were captured by forward hooks.")

        # Stage grid dimensions for swin_tiny_patch4_window7_224 (12 blocks)
        # Stage 0: 2 blocks (56x56 grid)
        # Stage 1: 2 blocks (28x28 grid)
        # Stage 2: 6 blocks (14x14 grid)
        # Stage 3: 2 blocks (7x7 grid)
        grid_sizes = [
            (56, 56), (56, 56),
            (28, 28), (28, 28),
            (14, 14), (14, 14), (14, 14), (14, 14), (14, 14), (14, 14),
            (7, 7), (7, 7),
        ]

        # Handle mismatch if model backbone varies
        if len(self.attn_maps) != len(grid_sizes):
            # Fallback estimation
            grid_sizes = [(7, 7)] * len(self.attn_maps)

        spatial_maps = []

        for idx, attn in enumerate(self.attn_maps):
            # Shape of attn: [num_windows, num_heads, 49, 49]
            H, W = grid_sizes[idx]
            num_win_h, num_win_w = H // 7, W // 7

            # Average attention weights across heads -> [num_windows, 49, 49]
            attn_heads = attn.mean(dim=1)

            # Add identity matrix for residual connection handling (Abnar & Zuidema, 2020)
            eye = torch.eye(49, device=attn.device).unsqueeze(0)
            attn_fused = 0.5 * attn_heads + 0.5 * eye
            attn_fused = attn_fused / torch.clamp(attn_fused.sum(dim=-1, keepdim=True), min=1e-6)

            # Token self-importance within window -> [num_windows, 49]
            tok_imp = attn_fused.mean(dim=1).reshape(num_win_h, num_win_w, 7, 7)

            # Reconstruct full 2D spatial grid for the stage -> [1, 1, H, W]
            spatial_grid = tok_imp.permute(0, 2, 1, 3).reshape(1, 1, H, W)

            # Interpolate spatial grid to input resolution [224, 224]
            resized_grid = F.interpolate(spatial_grid, size=(224, 224), mode="bilinear", align_corners=False)
            spatial_maps.append(resized_grid)

        # Accumulate attention maps across all 12 blocks via normalized mean
        rollout_stack = torch.stack(spatial_maps, dim=0)  # Shape: [12, 1, 1, 224, 224]
        rollout_map = rollout_stack.mean(dim=0)[0, 0].cpu().numpy()

        # Min-max normalization to [0, 1]
        r_min, r_max = rollout_map.min(), rollout_map.max()
        rollout_norm = (rollout_map - r_min) / max((r_max - r_min), 1e-6)

        return rollout_norm


def run_explainability_comparison(
    kaggle_checkpoint: str = "checkpoints/kaggle_best_model.pth",
    kaggle_config: str = "configs/kaggle_config.yaml",
    brats_checkpoint: str = "checkpoints/brats_best_model.pth",
    brats_config: str = "configs/brats_fusion_config.yaml",
    output_dir: str = "results/explainability_comparison",
):
    """
    Generates side-by-side explainability comparison figures (Original MRI | Grad-CAM | Attention Rollout)
    across Kaggle and BraTS test set samples.
    """
    print("=" * 85)
    print("=== DAY 25: ATTENTION ROLLOUT & SIDE-BY-SIDE EXPLAINABILITY COMPARISON ===")
    print("=" * 85)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}", flush=True)

    out_dir = PROJECT_ROOT / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_summary = []

    # -------------------------------------------------------------------------
    # PART 1: Kaggle Single-Modality Model Comparison
    # -------------------------------------------------------------------------
    print("\n>>> Running Comparison on Kaggle Single-Modality Test Set ...", flush=True)
    kaggle_model, k_cfg = load_model_from_config(kaggle_checkpoint, kaggle_config, device=device)
    k_class_names = k_cfg.dataset.class_names
    k_rollout = SwinAttentionRollout(kaggle_model, device=device)

    df_k = pd.read_csv(PROJECT_ROOT / "data/processed/kaggle_splits.csv")
    df_k_test = df_k[df_k["split"].str.lower() == "test"].reset_index(drop=True)

    # Sample 1 representative image per class for Kaggle
    k_samples = []
    for cls in k_class_names:
        sub = df_k_test[df_k_test["class_name"] == cls]
        if len(sub) > 0:
            k_samples.append(sub.iloc[0].to_dict())

    for idx, rec in enumerate(k_samples, start=1):
        rel_path = rec["file_path"]
        true_cls = rec["class_name"]
        npz_file = (PROJECT_ROOT / "data" / rel_path).resolve()
        if not npz_file.exists():
            npz_file = (PROJECT_ROOT / rel_path).resolve()

        if not npz_file.exists():
            continue

        raw_img = np.load(npz_file)["image"]
        img_tensor = pass_through_kaggle(raw_img, return_tensor=True)

        # 1. Grad-CAM
        g_cam, pred_idx, conf = generate_gradcam_heatmap(kaggle_model, img_tensor, device=device)
        pred_cls = k_class_names[pred_idx]
        is_corr = pred_cls == true_cls

        # 2. Attention Rollout
        r_cam = k_rollout.compute_rollout(img_tensor)

        # Display RGB Image
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_img = np.clip(img_tensor.numpy().transpose(1, 2, 0) * std + mean, 0.0, 1.0)

        _, g_blended = create_heatmap_overlay(rgb_img, g_cam, alpha=0.5)
        _, r_blended = create_heatmap_overlay(rgb_img, r_cam, alpha=0.5)

        # Plot 3-Panel Figure: [Original | Grad-CAM | Attention Rollout]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Original MRI Image\nTrue Class: {true_cls}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        color_str = "green" if is_corr else "red"
        axes[1].imshow(g_blended)
        axes[1].set_title(
            f"Grad-CAM Heatmap\nPred: {pred_cls} ({conf * 100:.1f}%)",
            fontsize=11,
            fontweight="bold",
            color=color_str,
        )
        axes[1].axis("off")

        axes[2].imshow(r_blended)
        axes[2].set_title(f"Attention Rollout Heatmap\n(Multi-Layer Rollout)", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        plt.suptitle(f"Kaggle Model Explainability Comparison | Sample #{idx} [{true_cls}]", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()

        out_fn = out_dir / f"comparison_kaggle_{idx:02d}_{true_cls}.png"
        plt.savefig(out_fn, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  [Kaggle {idx}/4] {true_cls:<11} | Pred: {pred_cls:<11} ({conf * 100:.1f}%) -> Saved: {out_fn.name}", flush=True)

        comparison_summary.append({
            "dataset": "Kaggle",
            "sample_id": idx,
            "true_label": true_cls,
            "pred_label": pred_cls,
            "confidence": conf,
            "filename": out_fn.name,
        })

    # -------------------------------------------------------------------------
    # PART 2: BraTS 4-Channel Fusion Model Comparison
    # -------------------------------------------------------------------------
    print("\n>>> Running Comparison on BraTS Multi-Modal Fusion Test Set ...", flush=True)
    brats_model, b_cfg = load_model_from_config(brats_checkpoint, brats_config, device=device)
    b_class_names = b_cfg.dataset.class_names
    b_rollout = SwinAttentionRollout(brats_model, device=device)

    df_b = pd.read_csv(PROJECT_ROOT / "data/processed/brats_splits.csv")
    df_b_test = df_b[df_b["split"].str.lower() == "test"].reset_index(drop=True)

    b_samples = []
    for grade in b_class_names:
        sub = df_b_test[df_b_test["grade"] == grade]
        for _, r in sub.iterrows():
            fp = (PROJECT_ROOT / "data" / r["file_path"]).resolve()
            if not fp.exists():
                fp = (PROJECT_ROOT / r["file_path"]).resolve()
            if fp.exists():
                data = np.load(fp)
                if np.sum(data["mask"]) > 50:
                    b_samples.append(r.to_dict())
                    break

    for idx, rec in enumerate(b_samples, start=1):
        rel_path = rec["file_path"]
        true_grade = rec["grade"]
        patient_id = rec.get("patient_id", "patient")

        npz_file = (PROJECT_ROOT / "data" / rel_path).resolve()
        if not npz_file.exists():
            npz_file = (PROJECT_ROOT / rel_path).resolve()

        if not npz_file.exists():
            continue

        npz_data = np.load(npz_file)
        t1 = npz_data["t1"]
        t1ce = npz_data["t1ce"]
        t2 = npz_data["t2"]
        flair = npz_data["flair"]
        mask = npz_data["mask"]

        fused_tensor = fuse_brats_modalities(t1, t1ce, t2, flair, return_tensor=True)

        # 1. Grad-CAM
        g_cam, pred_idx, conf = generate_gradcam_heatmap(brats_model, fused_tensor, device=device)
        pred_grade = b_class_names[pred_idx]
        is_corr = pred_grade == true_grade

        # 2. Attention Rollout
        r_cam = b_rollout.compute_rollout(fused_tensor)

        # FLAIR Background
        flair_min, flair_max = flair.min(), flair.max()
        flair_norm = (flair - flair_min) / max((flair_max - flair_min), 1e-6)

        _, g_blended = create_heatmap_overlay(flair_norm, g_cam, alpha=0.5)
        _, r_blended = create_heatmap_overlay(flair_norm, r_cam, alpha=0.5)

        # Plot 3-Panel Figure
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        axes[0].imshow(flair_norm, cmap="gray")
        if np.sum(mask) > 0:
            axes[0].contour(mask, levels=[0.5], colors=["magenta"], linewidths=1.8)
        axes[0].set_title(f"FLAIR MRI (Display Background)\nTrue Grade: {true_grade} (Mask in Magenta)", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        color_str = "green" if is_corr else "red"
        axes[1].imshow(g_blended)
        if np.sum(mask) > 0:
            axes[1].contour(mask, levels=[0.5], colors=["magenta"], linewidths=1.8)
        axes[1].set_title(
            f"Grad-CAM Heatmap\nPred: {pred_grade} ({conf * 100:.1f}%)",
            fontsize=10,
            fontweight="bold",
            color=color_str,
        )
        axes[1].axis("off")

        axes[2].imshow(r_blended)
        if np.sum(mask) > 0:
            axes[2].contour(mask, levels=[0.5], colors=["magenta"], linewidths=1.8)
        axes[2].set_title(f"Attention Rollout Heatmap\n(Multi-Layer Rollout)", fontsize=10, fontweight="bold")
        axes[2].axis("off")

        plt.suptitle(f"BraTS Model Explainability Comparison | {patient_id} [{true_grade}]", fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout()

        out_fn = out_dir / f"comparison_brats_{idx:02d}_{true_grade}_{patient_id}.png"
        plt.savefig(out_fn, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  [BraTS {idx}/2] {true_grade:<4} | Pred: {pred_grade:<4} ({conf * 100:.1f}%) -> Saved: {out_fn.name}", flush=True)

        comparison_summary.append({
            "dataset": "BraTS",
            "sample_id": idx,
            "true_label": true_grade,
            "pred_label": pred_grade,
            "confidence": conf,
            "filename": out_fn.name,
        })

    print("\n" + "=" * 85)
    print("=== EXPLAINABILITY COMPARISON PIPELINE COMPLETE ===")
    print(f"Generated {len(comparison_summary)} comparison figures in: {out_dir.resolve()}")
    print("=" * 85)

    return comparison_summary


def main():
    parser = argparse.ArgumentParser(description="Run Side-by-Side Explainability Comparison (Grad-CAM vs Attention Rollout)")
    parser.add_argument(
        "--kaggle_checkpoint",
        type=str,
        default="checkpoints/kaggle_best_model.pth",
        help="Path to Kaggle model checkpoint",
    )
    parser.add_argument(
        "--kaggle_config",
        type=str,
        default="configs/kaggle_config.yaml",
        help="Path to Kaggle config file",
    )
    parser.add_argument(
        "--brats_checkpoint",
        type=str,
        default="checkpoints/brats_best_model.pth",
        help="Path to BraTS model checkpoint",
    )
    parser.add_argument(
        "--brats_config",
        type=str,
        default="configs/brats_fusion_config.yaml",
        help="Path to BraTS config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/explainability_comparison",
        help="Output directory for comparison plots",
    )

    args = parser.parse_args()
    run_explainability_comparison(
        kaggle_checkpoint=args.kaggle_checkpoint,
        kaggle_config=args.kaggle_config,
        brats_checkpoint=args.brats_checkpoint,
        brats_config=args.brats_config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
