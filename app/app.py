"""
Streamlit Web Application for Explainable Brain Tumor Diagnosis.

Supports two diagnostic modes:
1. Single-Modality (Kaggle):
   - Single MRI scan upload (JPG/PNG)
   - 4-class classification (Glioma, Meningioma, No Tumor, Pituitary)
   - Swin-Tiny backbone with Grad-CAM explainability
2. Multi-Modal Fusion (BraTS):
   - 4-channel early fusion of co-registered MRI sequences (T1, T1c, T2, FLAIR)
   - Binary tumor grading (Low-Grade Glioma [LGG] vs. High-Grade Glioma [HGG])
   - Swin-Base backbone with dual Grad-CAM & Attention Rollout explainability
"""

import io
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attention_rollout import SwinAttentionRollout
from src.explain import create_heatmap_overlay, generate_gradcam_heatmap
from src.fusion.fusion import fuse_brats_modalities, pass_through_kaggle
from src.models.swin_model import SwinClassifier, build_swin_classifier
from src.preprocessing.normalize import (
    normalize_kaggle_image,
    resize_image,
    z_score_normalize_brats,
)
from src.utils.config_loader import load_config


# ==============================================================================
# CACHED MODEL LOADERS (WITH BACKBONE AUTO-DETECTION)
# ==============================================================================

@st.cache_resource(show_spinner="Loading Kaggle Swin-Tiny model weights...")
def load_kaggle_model(
    checkpoint_path: str = "checkpoints/kaggle_best_model.pth",
    config_path: str = "configs/kaggle_config.yaml",
) -> Tuple[SwinClassifier, Dict]:
    """
    Robustly loads the single-modality Kaggle model with backbone auto-detection.
    """
    ckpt_file = PROJECT_ROOT / checkpoint_path if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
    cfg_file = PROJECT_ROOT / config_path if not Path(config_path).is_absolute() else Path(config_path)

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found at: {cfg_file}")

    cfg = load_config(cfg_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)

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

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    meta = {
        "backbone_name": backbone_name,
        "input_channels": cfg.dataset.input_channels,
        "num_classes": cfg.dataset.num_classes,
        "class_names": cfg.dataset.class_names,
        "device": device,
    }
    return model, meta


@st.cache_resource(show_spinner="Loading BraTS Multi-Modal Swin-Base model weights...")
def load_brats_model(
    checkpoint_path: str = "checkpoints/brats_best_model.pth",
    config_path: str = "configs/brats_fusion_config.yaml",
) -> Tuple[SwinClassifier, Dict]:
    """
    Robustly loads the 4-channel BraTS fusion model with backbone auto-detection.
    """
    ckpt_file = PROJECT_ROOT / checkpoint_path if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
    cfg_file = PROJECT_ROOT / config_path if not Path(config_path).is_absolute() else Path(config_path)

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found at: {cfg_file}")

    cfg = load_config(cfg_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)

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

    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    meta = {
        "backbone_name": backbone_name,
        "input_channels": cfg.dataset.input_channels,
        "num_classes": cfg.dataset.num_classes,
        "class_names": cfg.dataset.class_names,
        "device": device,
    }
    return model, meta


# ==============================================================================
# PREPROCESSING HELPERS
# ==============================================================================

def preprocess_kaggle_image(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    Decodes and preprocesses a Kaggle single-modality MRI image strictly matching training:
    1. OpenCV byte decoding -> RGB uint8
    2. Resize to (224, 224) via resize_image
    3. ImageNet normalization via normalize_kaggle_image
    4. Tensor formatting via pass_through_kaggle
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Uploaded file could not be decoded as a valid image.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = resize_image(img_rgb, target_size=(224, 224), is_mask=False)
    display_bg = resized_rgb.astype(np.float32) / 255.0
    normalized = normalize_kaggle_image(resized_rgb, method="imagenet")
    image_tensor = pass_through_kaggle(normalized, target_size=(224, 224), return_tensor=True)

    return img_rgb, display_bg, image_tensor


# Backward compatibility alias for single-modality preprocessing
preprocess_uploaded_image = preprocess_kaggle_image


def decode_brats_modality_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Decodes raw file bytes for a BraTS modality slice (supporting PNG, JPG, and NPZ/NPY formats).
    """
    # Check if numpy array (.npy / .npz)
    if file_bytes.startswith(b"\x93NUMPY"):
        arr = np.load(io.BytesIO(file_bytes)).astype(np.float32)
    elif file_bytes.startswith(b"PK\x03\x04") or filename.endswith(".npz"):
        data = np.load(io.BytesIO(file_bytes))
        # Find key matching modality name or take first array
        key = list(data.keys())[0]
        for k in data.keys():
            if k in filename.lower():
                key = k
                break
        arr = data[key].astype(np.float32)
    else:
        # Standard image decoding
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            raise ValueError(f"Could not decode image file: {filename}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        arr = img.astype(np.float32)

    return arr


def preprocess_brats_slice(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses a single BraTS modality slice strictly matching training logic:
    1. Resize to (224, 224) via resize_image
    2. Intensity Z-score normalization over non-zero brain voxels via z_score_normalize_brats

    Returns:
        norm_slice (np.ndarray): Z-score normalized float32 slice (224, 224).
        display_slice (np.ndarray): Min-max scaled [0, 1] slice for visual rendering.
    """
    resized = resize_image(arr, target_size=(224, 224), is_mask=False)
    norm_slice = z_score_normalize_brats(resized, non_zero_only=True)

    # Scaled to [0, 1] for visual display
    s_min, s_max = resized.min(), resized.max()
    display_slice = (resized - s_min) / max((s_max - s_min), 1e-6)
    display_slice = np.clip(display_slice, 0.0, 1.0)

    return norm_slice, display_slice


# ==============================================================================
# MAIN STREAMLIT APP
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Explainable Brain Tumor Diagnosis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧠 Explainable Brain Tumor Diagnosis System")
    st.markdown(
        """
        An explainable deep learning platform for brain tumor diagnosis and grading from MRI scans, 
        powered by **Hierarchical Vision Transformers (Swin-Tiny & Swin-Base)** paired with 
        **Grad-CAM** and **Attention Rollout** visual interpretability.
        """
    )

    # Sidebar: Model status, benchmark specs, and clinical disclaimer
    st.sidebar.header("🔬 Diagnostic Systems & Metadata")

    st.sidebar.markdown(
        """
        ### 1. Kaggle Single-Modality
        - **Architecture:** `swin_tiny_patch4_window7_224`
        - **Input:** Single 2D MRI (3-channel stem)
        - **Test Accuracy:** **87.75%** | **AUC:** **0.9759**
        - **Classes:** `glioma`, `meningioma`, `notumor`, `pituitary`
        - **XAI:** Grad-CAM saliency

        ---

        ### 2. BraTS Multi-Modal Fusion
        - **Architecture:** `swin_base_patch4_window7_224`
        - **Input:** 4-Channel Early Fusion (`T1`, `T1c`, `T2`, `FLAIR`)
        - **Test Accuracy:** **86.73%** | **AUC:** **0.9677**
        - **Classes:** `LGG` (Low-Grade) vs `HGG` (High-Grade)
        - **XAI:** Dual Grad-CAM & Attention Rollout
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.warning(
        "⚠️ **Research & Educational Disclaimer**\n\n"
        "This repository and software are designed for **research and educational purposes only** "
        "and do not constitute a certified medical device. All diagnostic findings must be verified "
        "by qualified medical professionals."
    )

    # Top-Level Mode Selector
    st.markdown("---")
    tab_kaggle, tab_brats = st.tabs([
        "📷 Single-Modality Diagnosis (Kaggle)",
        "🧬 Multi-Modal Fusion Grading (BraTS)",
    ])

    # ==========================================================================
    # TAB 1: KAGGLE SINGLE-MODALITY MODE
    # ==========================================================================
    with tab_kaggle:
        st.subheader("Single-Modality 4-Class Brain Tumor Classification")
        st.markdown(
            "Upload a single 2D axial/coronal/sagittal MRI slice image. The model classifies the scan into "
            "**Glioma**, **Meningioma**, **No Tumor**, or **Pituitary** and renders a Grad-CAM activation overlay."
        )

        try:
            kaggle_model, kaggle_meta = load_kaggle_model()
            k_device = kaggle_meta["device"]
            k_classes = kaggle_meta["class_names"]
        except Exception as e:
            st.error(f"❌ Failed to load Kaggle model: {e}")
            return

        k_file = st.file_uploader(
            "Select an MRI slice image (JPEG or PNG format):",
            type=["jpg", "jpeg", "png"],
            key="kaggle_uploader",
            help="Upload a single MRI scan image.",
        )

        if k_file is None:
            st.info("👆 Please upload an MRI slice image to start single-modality analysis.")
        else:
            file_bytes = k_file.getvalue()
            try:
                raw_rgb, display_bg, img_tensor = preprocess_kaggle_image(file_bytes)
            except Exception as e:
                st.error(f"❌ **Invalid Image File:** {e}")
                return

            st.success(f"✅ Preprocessed `{k_file.name}` (Original size: {raw_rgb.shape[1]}x{raw_rgb.shape[0]})")

            # Inference
            with torch.no_grad():
                logits = kaggle_model(img_tensor.unsqueeze(0).to(k_device))
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                pred_class = k_classes[pred_idx]
                confidence = float(probs[pred_idx]) * 100.0

            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric("Predicted Diagnostic Class", pred_class.capitalize())
                st.metric("Model Confidence", f"{confidence:.2f}%")

            with res_col2:
                st.markdown("**Diagnostic Probability Distribution:**")
                prob_df = pd.DataFrame(
                    {
                        "Diagnostic Class": [c.capitalize() for c in k_classes],
                        "Probability (%)": probs * 100.0,
                    }
                ).set_index("Diagnostic Class")
                st.bar_chart(prob_df, y="Probability (%)")

                for cname, p in zip(k_classes, probs):
                    st.progress(float(p), text=f"{cname.capitalize()}: {p * 100.0:.2f}%")

            st.markdown("#### Visual Explainability (Grad-CAM Saliency)")
            with st.spinner("Generating Grad-CAM heatmap..."):
                g_cam, _, _ = generate_gradcam_heatmap(
                    model=kaggle_model,
                    image_tensor=img_tensor,
                    target_category=pred_idx,
                    device=k_device,
                )
                _, blended_overlay = create_heatmap_overlay(
                    gray_image=display_bg,
                    grayscale_cam=g_cam,
                    alpha=0.5,
                )

            col_img, col_cam = st.columns(2)
            with col_img:
                st.image(
                    display_bg,
                    caption=f"Original MRI Scan (Resized 224x224): {k_file.name}",
                    use_container_width=True,
                )
            with col_cam:
                st.image(
                    blended_overlay,
                    caption=f"Grad-CAM Overlay (Target: {pred_class.capitalize()} — {confidence:.2f}%)",
                    use_container_width=True,
                )

    # ==========================================================================
    # TAB 2: BRATS MULTI-MODAL FUSION MODE
    # ==========================================================================
    with tab_brats:
        st.subheader("Multi-Modal MRI Fusion Tumor Grading (BraTS)")
        st.markdown(
            "Upload all **FOUR** co-registered MRI sequence slices for the **same patient and slice position**: "
            "**T1** (anatomical baseline), **T1c** (contrast enhancement), **T2** (fluid/edema), and **FLAIR** (CSF suppression). "
            "The model stacks these into a 4-channel fused tensor `[4, 224, 224]` and predicts **High-Grade Glioma (HGG)** vs. "
            "**Low-Grade Glioma (LGG)**, generating both **Grad-CAM** and **Attention Rollout** visual explanations."
        )

        try:
            brats_model, brats_meta = load_brats_model()
            b_device = brats_meta["device"]
            b_classes = brats_meta["class_names"]
        except Exception as e:
            st.error(f"❌ Failed to load BraTS fusion model: {e}")
            return

        st.info("ℹ️ **Instruction:** All 4 uploaded scans must be from the exact same patient and anatomical slice level for valid multi-modal fusion.")

        # 4 File Uploaders in Grid Layout
        col_t1, col_t1c, col_t2, col_flair = st.columns(4)

        with col_t1:
            st.markdown("**1. T1 Native Scan**")
            file_t1 = st.file_uploader("Upload T1", type=["jpg", "jpeg", "png", "npz"], key="brats_t1")

        with col_t1c:
            st.markdown("**2. T1c Post-Contrast**")
            file_t1c = st.file_uploader("Upload T1c / T1ce", type=["jpg", "jpeg", "png", "npz"], key="brats_t1c")

        with col_t2:
            st.markdown("**3. T2 Fluid Scan**")
            file_t2 = st.file_uploader("Upload T2", type=["jpg", "jpeg", "png", "npz"], key="brats_t2")

        with col_flair:
            st.markdown("**4. FLAIR Scan**")
            file_flair = st.file_uploader("Upload FLAIR", type=["jpg", "jpeg", "png", "npz"], key="brats_flair")

        # Input Validation: Check that all 4 modalities are present
        uploaded_files = [file_t1, file_t1c, file_t2, file_flair]
        uploaded_count = sum(1 for f in uploaded_files if f is not None)

        if uploaded_count == 0:
            st.info("👆 Please upload all 4 MRI modalities above to begin multi-modal analysis.")
        elif uploaded_count < 4:
            missing = []
            if file_t1 is None: missing.append("T1")
            if file_t1c is None: missing.append("T1c")
            if file_t2 is None: missing.append("T2")
            if file_flair is None: missing.append("FLAIR")
            st.warning(
                f"⚠️ **Incomplete Input:** Please upload all 4 modalities to proceed ({uploaded_count}/4 uploaded). "
                f"Missing modalities: **{', '.join(missing)}**."
            )
        else:
            # Preprocess all 4 modalities
            try:
                arr_t1 = decode_brats_modality_bytes(file_t1.getvalue(), file_t1.name)
                arr_t1c = decode_brats_modality_bytes(file_t1c.getvalue(), file_t1c.name)
                arr_t2 = decode_brats_modality_bytes(file_t2.getvalue(), file_t2.name)
                arr_flair = decode_brats_modality_bytes(file_flair.getvalue(), file_flair.name)

                norm_t1, disp_t1 = preprocess_brats_slice(arr_t1)
                norm_t1c, disp_t1c = preprocess_brats_slice(arr_t1c)
                norm_t2, disp_t2 = preprocess_brats_slice(arr_t2)
                norm_flair, disp_flair = preprocess_brats_slice(arr_flair)

                # Early Fusion: Channel-stacking into [4, 224, 224] tensor
                fused_tensor = fuse_brats_modalities(
                    t1=norm_t1,
                    t1ce=norm_t1c,
                    t2=norm_t2,
                    flair=norm_flair,
                    target_size=(224, 224),
                    return_tensor=True,
                )
            except Exception as e:
                st.error(f"❌ **Error preprocessing BraTS modalities:** {e}")
                return

            st.success("✅ Successfully preprocessed and fused all 4 modalities into a [4, 224, 224] tensor!")

            # Display 4 Uploaded Modalities Grid
            st.markdown("#### 1. Fused MRI Sequence Scans (Input Channels 0–3)")
            disp_c1, disp_c2, disp_c3, disp_c4 = st.columns(4)
            with disp_c1:
                st.image(disp_t1, caption="Channel 0: T1 Native", use_container_width=True)
            with disp_c2:
                st.image(disp_t1c, caption="Channel 1: T1c Contrast", use_container_width=True)
            with disp_c3:
                st.image(disp_t2, caption="Channel 2: T2 Fluid", use_container_width=True)
            with disp_c4:
                st.image(disp_flair, caption="Channel 3: FLAIR (Baseline Display)", use_container_width=True)

            # Inference
            with torch.no_grad():
                input_batch = fused_tensor.unsqueeze(0).to(b_device)
                logits = brats_model(input_batch)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                pred_grade = b_classes[pred_idx]
                confidence = float(probs[pred_idx]) * 100.0

            grade_title = "High-Grade Glioma (HGG)" if pred_grade == "HGG" else "Low-Grade Glioma (LGG)"

            # Diagnostic Metrics & Probability Breakdown
            st.markdown("---")
            st.markdown("#### 2. Diagnostic Grade Prediction")
            b_col1, b_col2 = st.columns([1, 2])

            with b_col1:
                st.metric("Predicted Tumor Grade", grade_title)
                st.metric("Model Confidence", f"{confidence:.2f}%")

            with b_col2:
                st.markdown("**Grade Probability Breakdown:**")
                prob_b_df = pd.DataFrame(
                    {
                        "Tumor Grade": ["Low-Grade Glioma (LGG)", "High-Grade Glioma (HGG)"],
                        "Probability (%)": probs * 100.0,
                    }
                ).set_index("Tumor Grade")
                st.bar_chart(prob_b_df, y="Probability (%)")

                for cname, p in zip(b_classes, probs):
                    label = "High-Grade Glioma (HGG)" if cname == "HGG" else "Low-Grade Glioma (LGG)"
                    st.progress(float(p), text=f"{label}: {p * 100.0:.2f}%")

            # Dual Explainability Overlays (Grad-CAM & Attention Rollout)
            st.markdown("---")
            st.markdown("#### 3. Dual Visual Explainability (Overlaid on FLAIR Modality)")
            st.markdown(
                "Per clinical neuroradiology protocol, **FLAIR** is selected as the baseline anatomical display "
                "because it highlights peritumoral edema while suppressing cerebrospinal fluid signals. "
                "Below are the **Grad-CAM** gradient saliency overlay and the **Swin Attention Rollout** self-attention map."
            )

            with st.spinner("Computing Grad-CAM and Swin Attention Rollout heatmaps..."):
                try:
                    # 1. Grad-CAM
                    g_cam, _, _ = generate_gradcam_heatmap(
                        model=brats_model,
                        image_tensor=fused_tensor,
                        target_category=pred_idx,
                        device=b_device,
                    )
                    _, g_blended = create_heatmap_overlay(
                        gray_image=disp_flair,
                        grayscale_cam=g_cam,
                        alpha=0.5,
                    )

                    # 2. Attention Rollout
                    rollout_engine = SwinAttentionRollout(model=brats_model, device=b_device)
                    r_cam = rollout_engine.compute_rollout(fused_tensor)
                    _, r_blended = create_heatmap_overlay(
                        gray_image=disp_flair,
                        grayscale_cam=r_cam,
                        alpha=0.5,
                    )
                except Exception as ex_err:
                    st.error(f"❌ Explainability generation failed: {ex_err}")
                    return

            exp_c1, exp_c2, exp_c3 = st.columns(3)
            with exp_c1:
                st.image(disp_flair, caption="Representative FLAIR Modality", use_container_width=True)
            with exp_c2:
                st.image(g_blended, caption=f"Grad-CAM Saliency Overlay ({pred_grade} — {confidence:.1f}%)", use_container_width=True)
            with exp_c3:
                st.image(r_blended, caption="Swin Attention Rollout Overlay", use_container_width=True)

    # Footer note
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 0.85em;'>"
        "Explainable Brain Tumor Diagnosis Platform | Swin Transformer Multi-Modal Fusion & XAI | "
        "Research and educational tool only — not for clinical diagnostic use."
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
