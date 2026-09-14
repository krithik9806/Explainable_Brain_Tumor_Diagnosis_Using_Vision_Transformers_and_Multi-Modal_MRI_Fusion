"""
Streamlit Web Application for Explainable Brain Tumor Diagnosis.

Redesigned Product-Grade Interface:
- Theme System: Light theme default with seamless Dark theme toggle.
- Clean SaaS Flow: Hero -> Visual Mode Selection -> Primary Upload Dropzone -> Analysis -> Explainability.
- Preloaded clinical samples demoted to secondary 1-click test expanders.
- Model benchmark stat cards moved to collapsible technical details section.
"""

import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
# PRELOADED CLINICAL DEMO CASES (SECONDARY QUICK-TEST DATA)
# ==============================================================================

KAGGLE_DEMO_SAMPLES = {
    "Glioma (Intra-Axial Lesion)": {
        "path": PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "glioma" / "Te-gl_1.jpg",
        "label": "Glioma",
        "description": "High-grade intra-axial parenchymal mass with mass effect.",
    },
    "Pituitary Adenoma (Sellar Mass)": {
        "path": PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "pituitary" / "Te-pi_1.jpg",
        "label": "Pituitary",
        "description": "Circumscribed sellar / parasellar adenoma.",
    },
    "Healthy Normal Control": {
        "path": PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "notumor" / "Te-no_1.jpg",
        "label": "No Tumor",
        "description": "Normal cerebral anatomical baseline without lesion or edema.",
    },
    "Meningioma (Dural-Based Mass)": {
        "path": PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "meningioma" / "Te-aug-me_1.jpg",
        "label": "Meningioma",
        "description": "Extra-axial, dural-based mass compressing adjacent cortical tissue.",
    },
}

BRATS_DEMO_SAMPLES = {
    "Patient 310 (Low-Grade Glioma / LGG, Slice 52)": {
        "path": PROJECT_ROOT / "data" / "processed" / "brats_normalized" / "BraTS20_Training_310" / "slice_052.npz",
        "grade": "LGG",
        "description": "WHO Grade II astrocytoma with hyperintense T2/FLAIR signal without aggressive necrosis.",
    },
    "Patient 020 (High-Grade Glioma / HGG, Slice 40)": {
        "path": PROJECT_ROOT / "data" / "processed" / "brats_normalized" / "BraTS20_Training_020" / "slice_040.npz",
        "grade": "HGG",
        "description": "WHO Grade IV glioblastoma displaying intense ring enhancement on T1ce and vasogenic edema.",
    },
    "Patient 006 (High-Grade Glioblastoma / HGG, Slice 84)": {
        "path": PROJECT_ROOT / "data" / "processed" / "brats_normalized" / "BraTS20_Training_006" / "slice_084.npz",
        "grade": "HGG",
        "description": "Large necrotic cavitary glioblastoma with marked ventricular compression.",
    },
}


# ==============================================================================
# CACHED MODEL LOADERS (WITH BACKBONE AUTO-DETECTION)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_kaggle_model(
    checkpoint_path: str = "checkpoints/kaggle_best_model.pth",
    config_path: str = "configs/kaggle_config.yaml",
) -> Tuple[SwinClassifier, Dict]:
    ckpt_file = PROJECT_ROOT / checkpoint_path if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
    cfg_file = PROJECT_ROOT / config_path if not Path(config_path).is_absolute() else Path(config_path)

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found at: {cfg_file}")

    cfg = load_config(cfg_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    backbone_name = checkpoint.get("backbone", cfg.model.backbone) if isinstance(checkpoint, dict) else cfg.model.backbone

    model = build_swin_classifier(
        backbone_name=backbone_name,
        input_channels=cfg.dataset.input_channels,
        num_classes=cfg.dataset.num_classes,
        pretrained=False,
    )

    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
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


@st.cache_resource(show_spinner=False)
def load_brats_model(
    checkpoint_path: str = "checkpoints/brats_best_model.pth",
    config_path: str = "configs/brats_fusion_config.yaml",
) -> Tuple[SwinClassifier, Dict]:
    ckpt_file = PROJECT_ROOT / checkpoint_path if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
    cfg_file = PROJECT_ROOT / config_path if not Path(config_path).is_absolute() else Path(config_path)

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found at: {cfg_file}")

    cfg = load_config(cfg_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    backbone_name = checkpoint.get("backbone", cfg.model.backbone) if isinstance(checkpoint, dict) else cfg.model.backbone

    model = build_swin_classifier(
        backbone_name=backbone_name,
        input_channels=cfg.dataset.input_channels,
        num_classes=cfg.dataset.num_classes,
        pretrained=False,
    )

    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
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

preprocess_uploaded_image = preprocess_kaggle_image


def decode_brats_modality_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    if file_bytes.startswith(b"\x93NUMPY"):
        return np.load(io.BytesIO(file_bytes)).astype(np.float32)
    elif file_bytes.startswith(b"PK\x03\x04") or filename.endswith(".npz"):
        data = np.load(io.BytesIO(file_bytes))
        key = list(data.keys())[0]
        for k in data.keys():
            if k in filename.lower():
                key = k
                break
        return data[key].astype(np.float32)
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None or img.size == 0:
            raise ValueError(f"Could not decode image file: {filename}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32)


def preprocess_brats_slice(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    resized = resize_image(arr, target_size=(224, 224), is_mask=False)
    norm_slice = z_score_normalize_brats(resized, non_zero_only=True)
    s_min, s_max = resized.min(), resized.max()
    display_slice = (resized - s_min) / max((s_max - s_min), 1e-6)
    display_slice = np.clip(display_slice, 0.0, 1.0)
    return norm_slice, display_slice


# ==============================================================================
# DYNAMIC CSS THEME INJECTION
# ==============================================================================

def inject_theme_css(is_dark: bool = False):
    """
    Injects polished, clinical-grade CSS with light default and full dark support.
    """
    if is_dark:
        bg_main = "#0b1329"
        bg_card = "#131f37"
        bg_card_hover = "#1a2a4a"
        border_color = "#1e293b"
        text_primary = "#f8fafc"
        text_secondary = "#94a3b8"
        accent = "#38bdf8"
        accent_bg = "rgba(56, 189, 248, 0.12)"
        accent_border = "#0284c7"
        card_shadow = "0 8px 24px -4px rgba(0, 0, 0, 0.45)"
        dropzone_bg = "#0f172a"
    else:
        bg_main = "#f8fafc"
        bg_card = "#ffffff"
        bg_card_hover = "#f1f5f9"
        border_color = "#e2e8f0"
        text_primary = "#0f172a"
        text_secondary = "#475569"
        accent = "#0284c7"
        accent_bg = "rgba(2, 132, 199, 0.08)"
        accent_border = "#38bdf8"
        card_shadow = "0 8px 20px -2px rgba(15, 23, 42, 0.06)"
        dropzone_bg = "#ffffff"

    css = f"""
    <style>
    /* Global App Background & Font Settings */
    .stApp {{
        background-color: {bg_main};
        color: {text_primary};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}

    /* Main Container Padding */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    /* Hero Section */
    .hero-container {{
        text-align: center;
        padding: 1.5rem 1rem 1rem 1rem;
        margin-bottom: 1.5rem;
    }}
    .hero-badge {{
        display: inline-block;
        padding: 4px 14px;
        background: {accent_bg};
        border: 1px solid {accent};
        border-radius: 9999px;
        color: {accent};
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }}
    .hero-title {{
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: {text_primary};
        margin: 0 0 0.5rem 0;
    }}
    .hero-sub {{
        font-size: 1.05rem;
        color: {text_secondary};
        max-width: 760px;
        margin: 0 auto;
        line-height: 1.5;
    }}

    /* Mode Selection Cards */
    .mode-card {{
        background: {bg_card};
        border: 2px solid {border_color};
        border-radius: 14px;
        padding: 20px;
        text-align: left;
        box-shadow: {card_shadow};
        transition: all 0.2s ease-in-out;
        min-height: 170px;
    }}
    .mode-card-active {{
        border-color: {accent} !important;
        background: {accent_bg} !important;
    }}
    .mode-icon {{
        font-size: 1.8rem;
        margin-bottom: 8px;
    }}
    .mode-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {text_primary};
        margin-bottom: 4px;
    }}
    .mode-desc {{
        font-size: 0.88rem;
        color: {text_secondary};
        line-height: 1.4;
    }}
    .mode-badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        background: {accent};
        color: white;
        margin-top: 10px;
    }}

    /* Dropzone and Container Cards */
    .upload-card {{
        background: {bg_card};
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: 24px;
        box-shadow: {card_shadow};
        margin-bottom: 1.5rem;
    }}
    .upload-card-header {{
        font-size: 1.2rem;
        font-weight: 700;
        color: {text_primary};
        margin-bottom: 4px;
    }}
    .upload-card-sub {{
        font-size: 0.88rem;
        color: {text_secondary};
        margin-bottom: 16px;
    }}

    /* Result Card Styling */
    .result-card {{
        background: {bg_card};
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: 24px;
        box-shadow: {card_shadow};
        margin-top: 1.5rem;
    }}
    .result-grade {{
        font-size: 1.8rem;
        font-weight: 800;
        color: {accent};
    }}
    .confidence-meter {{
        background: {border_color};
        border-radius: 9999px;
        height: 10px;
        overflow: hidden;
        margin-top: 8px;
    }}
    .confidence-fill {{
        background: {accent};
        height: 100%;
        border-radius: 9999px;
    }}

    /* Dropzone grid item */
    .dropzone-box {{
        background: {dropzone_bg};
        border: 1px dashed {border_color};
        border-radius: 10px;
        padding: 10px;
        text-align: center;
    }}

    /* Button Polish */
    div.stButton > button {{
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
        transition: all 0.15s ease-in-out;
    }}
    div.stButton > button[kind="primary"] {{
        background-color: {accent};
        border: none;
        color: #ffffff;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background-color: {accent_border};
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {bg_card};
        border-right: 1px solid {border_color};
    }}

    /* Discreet Footer */
    .footer-text {{
        text-align: center;
        color: {text_secondary};
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid {border_color};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    st.set_page_config(
        page_title="NeuroVision | Explainable Brain Tumor Diagnosis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize Session State
    if "selected_mode" not in st.session_state:
        st.session_state["selected_mode"] = "single"  # 'single' (Kaggle) or 'multimodal' (BraTS)
    if "k_analysis_done" not in st.session_state:
        st.session_state["k_analysis_done"] = False
    if "b_analysis_done" not in st.session_state:
        st.session_state["b_analysis_done"] = False

    # 1. THEME TOGGLE (Sidebar - Light Default)
    with st.sidebar:
        st.markdown("### ⚙️ Display Settings")
        theme_choice = st.radio(
            "Theme Mode:",
            ["☀️ Light Theme (Default)", "🌙 Dark Theme"],
            index=0,
            key="theme_mode_radio",
        )
        is_dark = "Dark" in theme_choice
        st.markdown("---")
        st.markdown("### 🧠 Diagnostic Architectures")
        st.markdown(
            """
            - **Single-Modality:** `Swin-Tiny` (27.52M params)
              - 4-Class Triage: *87.75% Test Acc, 0.9759 AUC*
            - **Multi-Modal Fusion:** `Swin-Base` (86.75M params)
              - 4-Sequence Grading: *86.73% Test Acc, 0.9677 AUC*
              - *94.64% LGG Minority Sensitivity*
            """
        )
        st.markdown("---")
        st.caption("🔬 Software intended for research and educational validation.")

    # Inject Dynamic Theme CSS
    inject_theme_css(is_dark=is_dark)

    # 2. HERO SECTION
    st.markdown(
        """
        <div class="hero-container">
            <span class="hero-badge">Clinical AI Decision Support</span>
            <h1 class="hero-title">NeuroVision AI</h1>
            <p class="hero-sub">
                Explainable brain tumor diagnosis and histological grading from MRI scans, 
                powered by Hierarchical Vision Transformers and transparent pixel-level attention maps.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. INTERACTIVE MODE SELECTOR (Side-by-Side Clickable Cards)
    m_col1, m_col2 = st.columns(2)

    is_single_active = st.session_state["selected_mode"] == "single"
    is_multi_active = st.session_state["selected_mode"] == "multimodal"

    with m_col1:
        card_class = "mode-card mode-card-active" if is_single_active else "mode-card"
        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="mode-icon">📷</div>
                <div class="mode-title">1. Single-Modality Diagnosis</div>
                <div class="mode-desc">
                    Upload a single 2D axial MRI slice to classify into <b>Glioma</b>, <b>Meningioma</b>, 
                    <b>Pituitary Adenoma</b>, or <b>Healthy Control</b>.
                </div>
                <span class="mode-badge">Swin-Tiny • Grad-CAM • 87.75% Acc</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("👉 Select Single-Modality Mode", key="btn_select_single", use_container_width=True):
            st.session_state["selected_mode"] = "single"
            st.rerun()

    with m_col2:
        card_class = "mode-card mode-card-active" if is_multi_active else "mode-card"
        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="mode-icon">🧬</div>
                <div class="mode-title">2. Multi-Modal MRI Fusion</div>
                <div class="mode-desc">
                    Early-fusion of 4 co-registered sequences (<b>T1, T1ce, T2, FLAIR</b>) for fine-grained 
                    grading of <b>Low-Grade (LGG)</b> vs. <b>High-Grade Gliomas (HGG)</b>.
                </div>
                <span class="mode-badge">Swin-Base • Dual XAI • 86.73% Acc</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("👉 Select Multi-Modal Fusion Mode", key="btn_select_multi", use_container_width=True):
            st.session_state["selected_mode"] = "multimodal"
            st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)

    # ==========================================================================
    # WORKFLOW A: SINGLE-MODALITY DIAGNOSIS (KAGGLE)
    # ==========================================================================
    if st.session_state["selected_mode"] == "single":
        st.markdown(
            """
            <div class="upload-card">
                <div class="upload-card-header">📤 Step 1: Upload Brain MRI Scan</div>
                <div class="upload-card-sub">Drag and drop any standard 2D axial, coronal, or sagittal T1/T2 MRI slice image (JPEG or PNG format).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            kaggle_model, kaggle_meta = load_kaggle_model()
            k_device = kaggle_meta["device"]
            k_classes = kaggle_meta["class_names"]
        except Exception as e:
            st.error(f"❌ Could not load Kaggle Swin-Tiny model: {e}")
            return

        up_col1, up_col2 = st.columns([2, 1])
        with up_col1:
            k_uploaded_file = st.file_uploader(
                "Drop your brain MRI scan here",
                type=["jpg", "jpeg", "png"],
                key="kaggle_file_input",
                help="Accepts standard single MRI image files.",
            )

        active_k_bytes = None
        active_k_name = ""

        if k_uploaded_file is not None:
            active_k_bytes = k_uploaded_file.getvalue()
            active_k_name = k_uploaded_file.name
            with up_col2:
                st.image(active_k_bytes, caption=f"Uploaded: {active_k_name}", width=180)

        # Secondary Path: Preloaded Demo Case Expander
        with st.expander("💡 Or test with a pre-verified clinical sample (1-Click Demo)"):
            st.caption("Select an authenticated clinical test image to run the diagnostic pipeline immediately:")
            demo_cols = st.columns(4)
            for idx, (d_name, d_info) in enumerate(KAGGLE_DEMO_SAMPLES.items()):
                with demo_cols[idx]:
                    if st.button(f"⚡ {d_name.split(' ')[0]}", key=f"btn_k_demo_{idx}", use_container_width=True):
                        if d_info["path"].exists():
                            with open(d_info["path"], "rb") as f:
                                active_k_bytes = f.read()
                            active_k_name = d_name
                            st.session_state["k_active_demo_bytes"] = active_k_bytes
                            st.session_state["k_active_demo_name"] = active_k_name
                            st.rerun()

            if "k_active_demo_bytes" in st.session_state and active_k_bytes is None:
                active_k_bytes = st.session_state["k_active_demo_bytes"]
                active_k_name = st.session_state["k_active_demo_name"]
                with up_col2:
                    st.image(active_k_bytes, caption=f"Demo Sample: {active_k_name}", width=180)

        # Primary Action Button
        analyze_clicked = False
        if active_k_bytes is not None:
            st.markdown("<br/>", unsafe_allow_html=True)
            action_col1, action_col2 = st.columns([2, 1])
            with action_col1:
                analyze_clicked = st.button("🔍 Run Diagnostic Analysis & Explainability", type="primary", use_container_width=True)
            with action_col2:
                if st.button("🔄 Reset Scan", use_container_width=True):
                    if "k_active_demo_bytes" in st.session_state:
                        del st.session_state["k_active_demo_bytes"]
                    if "k_active_demo_name" in st.session_state:
                        del st.session_state["k_active_demo_name"]
                    st.rerun()

        # Run Inference and Display Results
        if active_k_bytes is not None and (analyze_clicked or st.session_state.get("k_auto_run", False)):
            with st.spinner("Processing MRI scan and computing Swin Transformer Grad-CAM heatmaps..."):
                try:
                    raw_rgb, display_bg, img_tensor = preprocess_kaggle_image(active_k_bytes)
                except Exception as e:
                    st.error(f"❌ Preprocessing failed: {e}")
                    return

                with torch.no_grad():
                    logits = kaggle_model(img_tensor.unsqueeze(0).to(k_device))
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    pred_idx = int(np.argmax(probs))
                    pred_class = k_classes[pred_idx]
                    confidence = float(probs[pred_idx]) * 100.0

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

            # Results Display Section
            st.markdown(
                f"""
                <div class="result-card">
                    <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">
                        Diagnostic Prediction
                    </div>
                    <div class="result-grade">{pred_class.capitalize()}</div>
                    <div style="font-size: 1.1rem; margin-top: 4px; color: #38bdf8;">
                        Model Confidence: <b>{confidence:.2f}%</b>
                    </div>
                    <div class="confidence-meter">
                        <div class="confidence-fill" style="width: {confidence:.1f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Probability Breakdown
            st.markdown("<br/>", unsafe_allow_html=True)
            res_c1, res_c2 = st.columns([1, 1])
            with res_c1:
                st.markdown("##### 📊 Class Probability Distribution")
                for cname, p in zip(k_classes, probs):
                    st.progress(float(p), text=f"{cname.capitalize()}: {p * 100.0:.2f}%")

            with res_c2:
                st.markdown("##### 🩺 Clinical Interpretation")
                if pred_class.lower() == "glioma":
                    st.info("Primary intra-axial parenchymal tumor. Recommend contrast-enhanced follow-up to evaluate microvascular proliferation.")
                elif pred_class.lower() == "meningioma":
                    st.info("Extra-axial dural-based lesion. Recommend neurosurgical assessment of dural tail and mass effect on cortex.")
                elif pred_class.lower() == "pituitary":
                    st.info("Circumscribed mass localized to the sella turcica. Recommend endocrine hormone panel and visual field evaluation.")
                else:
                    st.success("No focal mass effect, midline shift, or gross intracranial signal anomaly detected.")

            # Heatmaps
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("##### 🎯 Visual Explainability (Grad-CAM Saliency)")
            g_c1, g_c2 = st.columns(2)
            with g_c1:
                st.image(display_bg, caption=f"Original Axial MRI Slice (224x224): {active_k_name}", use_container_width=True)
            with g_c2:
                st.image(blended_overlay, caption=f"Grad-CAM Heatmap ({pred_class.capitalize()} — {confidence:.1f}%)", use_container_width=True)

        # Collapsible Technical Model Specs (Moved OUT of primary flow)
        with st.expander("ℹ️ About the Swin-Tiny Model & Benchmark Specifications"):
            st.markdown(
                """
                - **Backbone Architecture:** `swin_tiny_patch4_window7_224` (27,516,548 parameters)
                - **Dataset:** Kaggle 4-Class Brain Tumor MRI Dataset (7,200 total images)
                - **Held-Out Test Set:** 1,600 balanced images (400 per class)
                - **Test Accuracy:** **87.75%** | **Macro F1:** **0.8751** | **ROC-AUC:** **0.9759 (OvR)**
                - **Explainability Target:** Stage 4 final normalization layer (`layers[-1].blocks[-1].norm2`)
                """
            )

    # ==========================================================================
    # WORKFLOW B: MULTI-MODAL MRI FUSION (BRATS)
    # ==========================================================================
    else:
        st.markdown(
            """
            <div class="upload-card">
                <div class="upload-card-header">📤 Step 1: Upload 4 Co-Registered MRI Sequences</div>
                <div class="upload-card-sub">Upload all 4 sequence slices from the exact same patient and slice position to perform multi-parametric early fusion.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            brats_model, brats_meta = load_brats_model()
            b_device = brats_meta["device"]
            b_classes = brats_meta["class_names"]
        except Exception as e:
            st.error(f"❌ Could not load BraTS Swin-Base model: {e}")
            return

        # 4 Dropzones Grid
        u_t1, u_t1c, u_t2, u_flair = st.columns(4)

        with u_t1:
            st.markdown("<div class='dropzone-box'><b>1. T1 Native</b><br/><small>Anatomy & Boundaries</small></div>", unsafe_allow_html=True)
            file_t1 = st.file_uploader("Upload T1", type=["jpg", "png", "npz"], key="b_up_t1", label_visibility="collapsed")

        with u_t1c:
            st.markdown("<div class='dropzone-box'><b>2. T1c Contrast</b><br/><small>Active Vascular Rim</small></div>", unsafe_allow_html=True)
            file_t1c = st.file_uploader("Upload T1c", type=["jpg", "png", "npz"], key="b_up_t1c", label_visibility="collapsed")

        with u_t2:
            st.markdown("<div class='dropzone-box'><b>3. T2 Fluid</b><br/><small>Water Content & Edema</small></div>", unsafe_allow_html=True)
            file_t2 = st.file_uploader("Upload T2", type=["jpg", "png", "npz"], key="b_up_t2", label_visibility="collapsed")

        with u_flair:
            st.markdown("<div class='dropzone-box'><b>4. FLAIR</b><br/><small>CSF-Suppressed Edema</small></div>", unsafe_allow_html=True)
            file_flair = st.file_uploader("Upload FLAIR", type=["jpg", "png", "npz"], key="b_up_flair", label_visibility="collapsed")

        uploaded_list = [file_t1, file_t1c, file_t2, file_flair]
        uploaded_count = sum(1 for f in uploaded_list if f is not None)

        fused_tensor = None
        disp_t1 = disp_t1c = disp_t2 = disp_flair = None
        active_b_name = ""

        # Status badge for uploads
        if uploaded_count == 4:
            st.success("✅ All 4 sequences uploaded! Ready for multi-modal early fusion.")
            try:
                arr_t1 = decode_brats_modality_bytes(file_t1.getvalue(), file_t1.name)
                arr_t1c = decode_brats_modality_bytes(file_t1c.getvalue(), file_t1c.name)
                arr_t2 = decode_brats_modality_bytes(file_t2.getvalue(), file_t2.name)
                arr_flair = decode_brats_modality_bytes(file_flair.getvalue(), file_flair.name)

                norm_t1, disp_t1 = preprocess_brats_slice(arr_t1)
                norm_t1c, disp_t1c = preprocess_brats_slice(arr_t1c)
                norm_t2, disp_t2 = preprocess_brats_slice(arr_t2)
                norm_flair, disp_flair = preprocess_brats_slice(arr_flair)

                fused_tensor = fuse_brats_modalities(
                    t1=norm_t1, t1ce=norm_t1c, t2=norm_t2, flair=norm_flair,
                    target_size=(224, 224), return_tensor=True
                )
                active_b_name = "User Multi-Modal Upload"
            except Exception as ex:
                st.error(f"❌ Error processing uploaded sequences: {ex}")
        elif uploaded_count > 0:
            st.info(f"⏳ {uploaded_count}/4 sequences uploaded. Please upload all 4 modalities to proceed.")

        # Secondary Path: Preloaded Multi-Modal Cohort Expander
        with st.expander("💡 Or test with a pre-verified multi-modal patient case (1-Click Demo)"):
            st.caption("Select an authenticated 4-sequence BraTS cohort subject to fuse and evaluate immediately:")
            b_demo_cols = st.columns(3)
            for idx, (b_dname, b_dinfo) in enumerate(BRATS_DEMO_SAMPLES.items()):
                with b_demo_cols[idx]:
                    if st.button(f"⚡ {b_dname.split(' ')[1]}", key=f"btn_b_demo_{idx}", use_container_width=True):
                        if b_dinfo["path"].exists():
                            data = np.load(b_dinfo["path"])
                            t1, t1ce, t2, flair = data["t1"], data["t1ce"], data["t2"], data["flair"]
                            norm_t1, disp_t1 = preprocess_brats_slice(t1)
                            norm_t1c, disp_t1c = preprocess_brats_slice(t1ce)
                            norm_t2, disp_t2 = preprocess_brats_slice(t2)
                            norm_flair, disp_flair = preprocess_brats_slice(flair)

                            fused_tensor = fuse_brats_modalities(
                                t1=norm_t1, t1ce=norm_t1c, t2=norm_t2, flair=norm_flair,
                                target_size=(224, 224), return_tensor=True
                            )
                            active_b_name = b_dname
                            st.session_state["b_fused_tensor"] = fused_tensor
                            st.session_state["b_disp_tuple"] = (disp_t1, disp_t1c, disp_t2, disp_flair)
                            st.session_state["b_active_name"] = active_b_name
                            st.session_state["b_active_desc"] = b_dinfo["description"]
                            st.session_state["b_true_grade"] = b_dinfo["grade"]
                            st.rerun()

            if "b_fused_tensor" in st.session_state and fused_tensor is None:
                fused_tensor = st.session_state["b_fused_tensor"]
                disp_t1, disp_t1c, disp_t2, disp_flair = st.session_state["b_disp_tuple"]
                active_b_name = st.session_state["b_active_name"]

        # Primary Action Button
        b_analyze_clicked = False
        if fused_tensor is not None:
            st.markdown("<br/>", unsafe_allow_html=True)
            b_act1, b_act2 = st.columns([2, 1])
            with b_act1:
                b_analyze_clicked = st.button("🧬 Fuse 4 Sequences & Diagnose Tumor Grade", type="primary", use_container_width=True)
            with b_act2:
                if st.button("🔄 Reset Sequences", use_container_width=True):
                    for k in ["b_fused_tensor", "b_disp_tuple", "b_active_name", "b_active_desc", "b_true_grade"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()

        # Run Inference and Display Results
        if fused_tensor is not None and (b_analyze_clicked or st.session_state.get("b_auto_run", False)):
            with st.spinner("Executing Swin-Base early fusion, Grad-CAM backpropagation, and Attention Rollout..."):
                with torch.no_grad():
                    logits = brats_model(fused_tensor.unsqueeze(0).to(b_device))
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    pred_idx = int(np.argmax(probs))
                    pred_grade = b_classes[pred_idx]
                    confidence = float(probs[pred_idx]) * 100.0

                grade_title = "High-Grade Glioma (HGG)" if pred_grade == "HGG" else "Low-Grade Glioma (LGG)"

                # Saliency Overlays
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

                rollout_engine = SwinAttentionRollout(model=brats_model, device=b_device)
                r_cam = rollout_engine.compute_rollout(fused_tensor)
                _, r_blended = create_heatmap_overlay(
                    gray_image=disp_flair,
                    grayscale_cam=r_cam,
                    alpha=0.5,
                )

            # Results Hero Card
            st.markdown(
                f"""
                <div class="result-card">
                    <div style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">
                        Histological Grade Prediction
                    </div>
                    <div class="result-grade">{grade_title}</div>
                    <div style="font-size: 1.1rem; margin-top: 4px; color: #38bdf8;">
                        Model Confidence: <b>{confidence:.2f}%</b>
                    </div>
                    <div class="confidence-meter">
                        <div class="confidence-fill" style="width: {confidence:.1f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Modality Inspection Grid
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("##### 🔬 Input 4-Channel Co-Registered MRI Sequences")
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.image(disp_t1, caption="T1 Native", use_container_width=True)
            with m2: st.image(disp_t1c, caption="T1c Contrast", use_container_width=True)
            with m3: st.image(disp_t2, caption="T2 Fluid", use_container_width=True)
            with m4: st.image(disp_flair, caption="FLAIR (CSF Suppressed)", use_container_width=True)

            # Dual Explainability Grid
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("##### 🎯 Dual Visual Explainability (Overlaid on Anatomical FLAIR)")
            st.caption("FLAIR suppresses free-water CSF to accentuate peritumoral edema. Grad-CAM isolates focal gradient activations, while Attention Rollout maps global context.")

            ex1, ex2, ex3 = st.columns(3)
            with ex1:
                st.image(disp_flair, caption="Baseline FLAIR Sequence", use_container_width=True)
            with ex2:
                st.image(g_blended, caption=f"Grad-CAM Saliency ({pred_grade} — {confidence:.1f}%)", use_container_width=True)
            with ex3:
                st.image(r_blended, caption="Swin Attention Rollout (Attention Flow)", use_container_width=True)

        # Collapsible Technical Model Specs (Moved OUT of primary flow)
        with st.expander("ℹ️ About the Swin-Base Fusion Model & BraTS Benchmark Specifications"):
            st.markdown(
                """
                - **Backbone Architecture:** `swin_base_patch4_window7_224` (86,746,478 parameters)
                - **Input Representation:** 4-Channel early fusion (`T1`, `T1ce`, `T2`, `FLAIR`) $[4, 224, 224]$
                - **Held-Out Test Set:** 588 patient-level slices (112 LGG, 476 HGG)
                - **Test Accuracy:** **86.73%** | **ROC-AUC:** **0.9677** | **HGG F1:** **0.9120**
                - **Minority LGG Sensitivity:** **94.64%** (Rescued from 0.00% majority-class collapse via loss reweighting & balanced mini-batches)
                """
            )

    # 4. DISCREET FOOTER
    st.markdown(
        """
        <div class="footer-text">
            NeuroVision AI • Explainable Brain Tumor Diagnosis Platform • MIT License<br/>
            Intended strictly for scientific research and educational evaluation. Not a certified clinical device.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
