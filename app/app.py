"""
Streamlit Interactive Web Application Demo for Brain Tumor Diagnosis & Visual Explainability (XAI).

Features:
1. Multi-Model Architecture Auto-Detection: Automatically detects backbone ('swin_tiny_patch4_window7_224' or
   'swin_base_patch4_window7_224') directly from PyTorch checkpoint dict metadata.
2. Supports Kaggle (Single-Modality 3-Channel 4-Class) and BraTS (Multi-Modal 4-Channel Fusion Binary HGG/LGG).
3. Real-Time Inference: Computes prediction logits, softmax probability distribution, and target class label.
4. Visual Explainability Overlays: Renders dual Grad-CAM and Attention Rollout heatmaps over MRI slices.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import torch
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.swin_model import build_swin_classifier, SwinClassifier
from src.utils.config_loader import load_config
from src.fusion.fusion import fuse_brats_modalities, pass_through_kaggle
from src.explain import generate_gradcam_heatmap, create_heatmap_overlay
from src.attention_rollout import SwinAttentionRollout


@st.cache_resource
def load_model_auto_detect(
    checkpoint_path: str,
    config_path: str,
) -> Tuple[SwinClassifier, Dict]:
    """
    Robustly loads model weights with automatic backbone architecture detection from checkpoint metadata.
    Prevents shape mismatch errors between swin_tiny and swin_base models.
    """
    ckpt_file = PROJECT_ROOT / checkpoint_path if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
    cfg_file = PROJECT_ROOT / config_path if not Path(config_path).is_absolute() else Path(config_path)

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")

    cfg = load_config(cfg_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)

    # AUTO-DETECTION FIX: Extract exact backbone architecture saved in checkpoint dict if available
    backbone_name = (
        checkpoint.get("backbone")
        if isinstance(checkpoint, dict) and "backbone" in checkpoint
        else cfg.model.backbone
    )

    model = build_swin_classifier(
        backbone_name=backbone_name,
        input_channels=cfg.dataset.input_channels,
        num_classes=cfg.dataset.num_classes,
        pretrained=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    meta = {
        "backbone_name": backbone_name,
        "input_channels": cfg.dataset.input_channels,
        "num_classes": cfg.dataset.num_classes,
        "class_names": cfg.dataset.class_names,
        "experiment_name": cfg.experiment_name,
    }
    return model, meta


def main():
    st.set_page_config(
        page_title="Explainable Brain Tumor Diagnosis UI",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 Explainable Brain Tumor Diagnosis System")
    st.markdown(
        "**Multi-Modal Vision Transformer (Swin-Base/Swin-Tiny) with Grad-CAM & Attention Rollout Interpretability**"
    )
    st.sidebar.header("⚙️ Experiment Selection & Settings")

    exp_choice = st.sidebar.selectbox(
        "Select Active Experiment Model:",
        options=[
            "BraTS 4-Channel Fusion (Swin-Base: HGG vs LGG)",
            "Kaggle Single-Modality (Swin-Tiny: 4-Class Classification)",
        ],
    )

    if "BraTS" in exp_choice:
        ckpt_path = "checkpoints/brats_best_model.pth"
        cfg_path = "configs/brats_fusion_config.yaml"
    else:
        ckpt_path = "checkpoints/kaggle_best_model.pth"
        cfg_path = "configs/kaggle_config.yaml"

    try:
        model, meta = load_model_auto_detect(ckpt_path, cfg_path)
        st.sidebar.success(
            f"Loaded Model: `{meta['backbone_name']}` ({meta['input_channels']} channels, {meta['num_classes']} classes)"
        )
    except Exception as e:
        st.sidebar.error(f"Failed to load checkpoint `{ckpt_path}`: {e}")
        return

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Auto-Detection Summary:**\n"
        f"- **Checkpoint Path**: `{ckpt_path}`\n"
        f"- **Backbone Architecture**: `{meta['backbone_name']}`\n"
        f"- **Configured Classes**: `{meta['class_names']}`"
    )

    st.subheader("1. Sample MRI Selection & Inspection")

    if "BraTS" in exp_choice:
        sample_dir = PROJECT_ROOT / "data" / "processed" / "brats"
        sample_files = list(sample_dir.glob("*.npz")) if sample_dir.exists() else []
    else:
        sample_dir = PROJECT_ROOT / "data" / "processed" / "kaggle"
        sample_files = list(sample_dir.glob("*.npz")) if sample_dir.exists() else []

    if not sample_files:
        st.warning(f"No processed `.npz` MRI slice samples found in `{sample_dir}`.")
        return

    selected_sample_path = st.selectbox(
        "Choose an MRI slice sample from held-out dataset:",
        options=[str(p.relative_to(PROJECT_ROOT)) for p in sample_files[:20]],
    )

    if selected_sample_path:
        npz_data = np.load(PROJECT_ROOT / selected_sample_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "BraTS" in exp_choice:
            t1, t1ce, t2, flair = npz_data["t1"], npz_data["t1ce"], npz_data["t2"], npz_data["flair"]
            mask = npz_data["mask"] if "mask" in npz_data else None
            img_tensor = fuse_brats_modalities(t1, t1ce, t2, flair, return_tensor=True)

            flair_norm = (flair - flair.min()) / max((flair.max() - flair.min()), 1e-6)
            display_bg = flair_norm
        else:
            raw_img = npz_data["image"] if "image" in npz_data else npz_data[list(npz_data.keys())[0]]
            img_tensor = pass_through_kaggle(raw_img, return_tensor=True)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            display_bg = np.clip(img_tensor.numpy().transpose(1, 2, 0) * std + mean, 0.0, 1.0)

        # Inference
        with torch.no_grad():
            input_batch = img_tensor.unsqueeze(0).to(device)
            logits = model(input_batch)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_class = meta["class_names"][pred_idx]
            confidence = probs[pred_idx]

        st.subheader("2. Model Prediction & Probability Distribution")
        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            st.metric("Predicted Diagnostic Class", pred_class)
            st.metric("Model Confidence", f"{confidence * 100:.2f}%")

        with col_res2:
            st.markdown("**Class Probability Breakdown:**")
            for idx, cname in enumerate(meta["class_names"]):
                st.progress(float(probs[idx]), text=f"{cname}: {probs[idx] * 100:.2f}%")

        st.subheader("3. Visual Explainability (Grad-CAM & Attention Rollout)")

        # Generate Explainability Overlays
        g_cam, _, _ = generate_gradcam_heatmap(model, img_tensor, device=device)
        rollout_engine = SwinAttentionRollout(model, device=device)
        r_cam = rollout_engine.compute_rollout(img_tensor)

        _, g_blended = create_heatmap_overlay(display_bg, g_cam, alpha=0.5)
        _, r_blended = create_heatmap_overlay(display_bg, r_cam, alpha=0.5)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(display_bg, caption="Input MRI Scan (Display Background)", use_column_width=True)
        with c2:
            st.image(g_blended, caption=f"Grad-CAM Heatmap (Pred: {pred_class})", use_column_width=True)
        with c3:
            st.image(r_blended, caption="Attention Rollout Heatmap", use_column_width=True)


if __name__ == "__main__":
    main()
