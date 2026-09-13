"""
Streamlit Web Application for Explainable Brain Tumor Diagnosis (Single-Modality MRI).

This interactive application allows clinicians and researchers to:
1. Upload a single 2D brain MRI scan (JPG/PNG).
2. Preprocess the image using the exact normalization and resizing pipeline used during model training.
3. Perform inference using the verified Swin Transformer model (checkpoints/kaggle_best_model.pth).
4. Inspect predicted diagnostic tumor classes, model confidence, and the full 4-class probability distribution.
5. Generate and visualize Grad-CAM saliency heatmaps side-by-side with the original scan.
"""

import io
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import create_heatmap_overlay, generate_gradcam_heatmap
from src.fusion.fusion import pass_through_kaggle
from src.models.swin_model import SwinClassifier, build_swin_classifier
from src.preprocessing.normalize import normalize_kaggle_image, resize_image
from src.utils.config_loader import load_config


@st.cache_resource(show_spinner="Loading trained Swin Transformer model...")
def load_kaggle_model(
    checkpoint_path: str = "checkpoints/kaggle_best_model.pth",
    config_path: str = "configs/kaggle_config.yaml",
) -> Tuple[SwinClassifier, Dict]:
    """
    Robustly loads the single-modality Kaggle model weights with automatic backbone architecture detection.
    
    Caches model in memory so it does not reload on each user interaction.
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

    # Backbone auto-detection (matching evaluate.py and explain.py)
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


def preprocess_uploaded_image(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    Decodes and preprocesses raw image bytes strictly using the training pipeline functions:
    1. OpenCV byte decoding -> RGB uint8 array
    2. Spatial resizing to (224, 224) via resize_image
    3. ImageNet intensity normalization via normalize_kaggle_image
    4. Tensor conversion into [3, 224, 224] via pass_through_kaggle

    Returns:
        raw_rgb (np.ndarray): Decoded RGB image array.
        display_bg (np.ndarray): Resized normalized RGB array in [0.0, 1.0] for display.
        image_tensor (torch.Tensor): Preprocessed float32 tensor of shape [3, 224, 224].

    Raises:
        ValueError: If bytes cannot be decoded as an image.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Uploaded file could not be decoded as a valid image.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Resize using src/preprocessing/normalize.py logic
    resized_rgb = resize_image(img_rgb, target_size=(224, 224), is_mask=False)

    # Display background in [0.0, 1.0]
    display_bg = resized_rgb.astype(np.float32) / 255.0

    # 2. Normalize using ImageNet statistics via src/preprocessing/normalize.py
    normalized = normalize_kaggle_image(resized_rgb, method="imagenet")

    # 3. Format into [3, 224, 224] torch.Tensor via src/fusion/fusion.py
    image_tensor = pass_through_kaggle(normalized, target_size=(224, 224), return_tensor=True)

    return img_rgb, display_bg, image_tensor


def main():
    st.set_page_config(
        page_title="Explainable Brain Tumor Diagnosis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Title and short overview description from README.md
    st.title("🧠 Explainable Brain Tumor Diagnosis System")
    st.markdown(
        """
        Brain tumors are among the most life-threatening neurological conditions, and early, accurate 
        diagnosis from MRI scans is critical for treatment planning. This application utilizes a 
        state-of-the-art **Hierarchical Swin Transformer** (`swin_tiny_patch4_window7_224`) to classify 
        brain MRI scans into 4 distinct categories (**Glioma**, **Meningioma**, **No Tumor**, and **Pituitary**) 
        and pairs every prediction with **Grad-CAM visual explainability** to reveal the anatomical regions driving the diagnosis.
        """
    )

    # Sidebar: Model status, metadata, and disclaimer
    st.sidebar.header("🔬 Model & System Information")
    st.sidebar.markdown(
        """
        - **Architecture:** `swin_tiny_patch4_window7_224`
        - **Input Modality:** Single 2D MRI (3-channel RGB stem)
        - **Input Resolution:** 224 × 224 pixels
        - **Test Accuracy:** **87.75%**
        - **Test AUC:** **0.9759** (Macro-Averaged)
        - **Target Classes:** `glioma`, `meningioma`, `notumor`, `pituitary`
        - **Explainability:** Grad-CAM (Shifted Window Token Reshape)
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.warning(
        "⚠️ **Research & Educational Disclaimer**\n\n"
        "This repository and application are intended for **research and educational purposes only** "
        "and do not constitute a certified clinical diagnostic system. Diagnostic decisions must always "
        "be confirmed by licensed radiologists and medical practitioners."
    )

    # Load Model with caching & backbone auto-detection
    try:
        model, meta = load_kaggle_model()
        device = meta["device"]
        class_names = meta["class_names"]
        st.sidebar.success(f"✅ Loaded weights: `{meta['backbone_name']}` on `{device}`")
    except Exception as e:
        st.error(f"❌ Failed to load model checkpoint: {e}")
        return

    st.markdown("---")
    st.subheader("📤 Upload Brain MRI Scan")
    uploaded_file = st.file_uploader(
        "Select a brain MRI slice image (JPEG or PNG format):",
        type=["jpg", "jpeg", "png"],
        help="Upload an axial, coronal, or sagittal T1/T2 brain MRI slice.",
    )

    if uploaded_file is None:
        st.info("👆 Please upload an MRI image to initiate classification and visual explainability.")
    else:
        # Preprocess with graceful error handling
        file_bytes = uploaded_file.getvalue()
        try:
            raw_rgb, display_bg, img_tensor = preprocess_uploaded_image(file_bytes)
        except Exception as e:
            st.error(
                f"❌ **Invalid Image File:** Unable to read or parse the uploaded file as a valid image ({e}). "
                "Please verify that the file is an intact JPG or PNG image."
            )
            return

        st.success(f"✅ Successfully loaded and preprocessed image: `{uploaded_file.name}` (Original dimensions: {raw_rgb.shape[1]}x{raw_rgb.shape[0]})")

        # Inference
        with torch.no_grad():
            input_batch = img_tensor.unsqueeze(0).to(device)
            logits = model(input_batch)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            confidence = float(probs[pred_idx]) * 100.0

        # Section 1: Diagnostic Prediction & Confidence
        st.markdown("---")
        st.subheader("🎯 Diagnostic Prediction & Probability Breakdown")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric(
                label="Predicted Class",
                value=pred_class.capitalize(),
            )
            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%",
            )

        with res_col2:
            st.markdown("**Diagnostic Probability Distribution:**")
            prob_df = pd.DataFrame(
                {
                    "Diagnostic Class": [c.capitalize() for c in class_names],
                    "Probability (%)": probs * 100.0,
                }
            ).set_index("Diagnostic Class")
            st.bar_chart(prob_df, y="Probability (%)")

            for cname, p in zip(class_names, probs):
                st.progress(float(p), text=f"{cname.capitalize()}: {p * 100.0:.2f}%")

        # Section 2: Visual Explainability (Grad-CAM)
        st.markdown("---")
        st.subheader("🔍 Visual Explainability (Grad-CAM)")
        st.markdown(
            "Grad-CAM visualizes the spatial attention of the Swin Transformer by projecting gradients "
            f"from the final attention layer onto the 2D feature map for class **{pred_class.capitalize()}**."
        )

        with st.spinner("Generating Grad-CAM attention heatmap..."):
            try:
                grayscale_cam, _, _ = generate_gradcam_heatmap(
                    model=model,
                    image_tensor=img_tensor,
                    target_category=pred_idx,
                    device=device,
                )
                _, blended_overlay = create_heatmap_overlay(
                    gray_image=display_bg,
                    grayscale_cam=grayscale_cam,
                    alpha=0.5,
                )
            except Exception as cam_err:
                st.error(f"❌ Grad-CAM generation failed: {cam_err}")
                return

        col_img, col_cam = st.columns(2)

        with col_img:
            st.image(
                display_bg,
                caption=f"Original MRI Scan (Resized 224x224): {uploaded_file.name}",
                use_container_width=True,
            )

        with col_cam:
            st.image(
                blended_overlay,
                caption=f"Grad-CAM Heatmap Overlay (Target: {pred_class.capitalize()} — {confidence:.2f}%)",
                use_container_width=True,
            )

    # Footer note
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray; font-size: 0.85em;'>"
        "Explainable Brain Tumor Diagnosis System | Built with PyTorch & Streamlit | "
        "Research and educational tool only — not for direct clinical diagnosis."
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
