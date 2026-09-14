"""
Streamlit Web Application for Explainable Brain Tumor Diagnosis.

Supports two primary diagnostic modes with preloaded clinical demo cases:
1. Single-Modality (Kaggle):
   - Preloaded clinical cases (Glioma, Meningioma, Pituitary, No Tumor) or custom upload
   - 4-class classification via Swin-Tiny (87.75% Test Acc, 0.9759 AUC)
   - Real-time Grad-CAM explainability overlay
2. Multi-Modal Fusion (BraTS):
   - Preloaded patient cases (LGG vs. HGG) or custom 4-channel upload
   - 4-channel early fusion of co-registered MRI sequences (T1, T1ce, T2, FLAIR)
   - Binary grading via Swin-Base (86.73% Test Acc, 0.9677 AUC, 94.64% LGG recall)
   - Dual Grad-CAM & Swin Attention Rollout visual explainability
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
# PRELOADED CLINICAL DEMO SAMPLE PATHS
# ==============================================================================

KAGGLE_DEMO_SAMPLES = {
    "Glioma (Intra-Axial Infiltrative Lesion)": {
        "path": PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "glioma" / "Te-gl_1.jpg",
        "true_label": "Glioma",
        "description": "Axial T1-weighted slice displaying an intra-axial heterogeneous parenchymal lesion with mass effect.",
    },
    "Pituitary Adenoma (Sellar / Parasellar Mass)": {
        "path": PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "pituitary" / "Te-pi_1.jpg",
        "true_label": "Pituitary",
        "description": "Axial slice localized to the sella turcica showing a circumscribed adenoma extending superiorly.",
    },
    "Healthy Brain (Healthy Normal Control)": {
        "path": PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "notumor" / "Te-no_1.jpg",
        "true_label": "No Tumor",
        "description": "Normal cerebral anatomical baseline without intracranial mass effect, shift, or signal anomaly.",
    },
    "Meningioma (Extra-Axial Dural-Based Mass)": {
        "path": PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "meningioma" / "Te-aug-me_1.jpg",
        "true_label": "Meningioma",
        "description": "Extra-axial, dural-based extra-parenchymal mass exerting compressive pressure on adjacent sulci.",
    },
}

BRATS_DEMO_SAMPLES = {
    "Patient 310 — Low-Grade Glioma (LGG, Slice 52)": {
        "path": PROJECT_ROOT / "data" / "processed" / "brats_normalized" / "BraTS20_Training_310" / "slice_052.npz",
        "true_grade": "LGG",
        "description": "WHO Grade II astrocytoma displaying hyperintense T2/FLAIR signal without aggressive necrotic cavitation or hypervascular enhancement.",
    },
    "Patient 020 — High-Grade Glioma (HGG, Slice 40)": {
        "path": PROJECT_ROOT / "data" / "processed" / "brats_normalized" / "BraTS20_Training_020" / "slice_040.npz",
        "true_grade": "HGG",
        "description": "WHO Grade IV Glioblastoma exhibiting prominent peripheral contrast enhancement on T1ce and severe vasogenic edema on FLAIR.",
    },
    "Patient 006 — High-Grade Glioblastoma (HGG, Slice 84)": {
        "path": PROJECT_ROOT / "data" / "processed" / "brats_normalized" / "BraTS20_Training_006" / "slice_084.npz",
        "true_grade": "HGG",
        "description": "Large necrotic cavitary glioblastoma with distinct infiltrative non-enhancing margins and substantial ventricular shift.",
    },
}


# ==============================================================================
# CACHED MODEL LOADERS (WITH BACKBONE AUTO-DETECTION)
# ==============================================================================

@st.cache_resource(show_spinner="Loading Kaggle Swin-Tiny model weights...")
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


@st.cache_resource(show_spinner="Loading BraTS Multi-Modal Swin-Base model weights...")
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
# MAIN STREAMLIT APP
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Explainable Brain Tumor Diagnosis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom Clean Clinical CSS
    st.markdown(
        """
        <style>
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 16px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        .metric-value {
            font-size: 2.1rem;
            font-weight: 700;
            color: #38bdf8;
            margin: 4px 0;
        }
        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 10px 18px;
            font-weight: 600;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            background: #0ea5e9;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # App Title & Header
    st.title("🧠 Explainable Brain Tumor Diagnosis Platform")
    st.markdown(
        "A clinical-grade deep learning system for multi-modal brain tumor classification and histological grading, "
        "powered by **Hierarchical Swin Transformers** and verified with **Grad-CAM** and **Attention Rollout** interpretability."
    )

    # Sidebar: Model metadata and verified results
    with st.sidebar:
        st.header("🔬 Clinical AI Models")
        st.markdown(
            """
            **1. Single-Modality Model**
            - Backbone: `Swin-Tiny` (27.52M params)
            - Benchmark: Kaggle 4-Class MRI
            - **Accuracy:** `87.75%` | **AUC:** `0.9759`
            - Classes: *Glioma, Meningioma, No Tumor, Pituitary*

            ---

            **2. Multi-Modal Fusion Model**
            - Backbone: `Swin-Base` (86.75M params)
            - Benchmark: BraTS 2020 Multi-Modal
            - **Accuracy:** `86.73%` | **AUC:** `0.9677`
            - **LGG Sensitivity:** `94.64%` (Recovered)
            - Input: 4-Channel early fusion (`T1, T1ce, T2, FLAIR`)
            """
        )
        st.markdown("---")
        st.info(
            "💡 **No empty uploads required:** Select any preloaded clinical case to test predictions and explainability heatmaps with one click!"
        )
        st.warning(
            "⚠️ **Research Tool Only:** Intended for scientific research and coursework evaluation; not certified for clinical diagnostic use."
        )

    # Top KPI Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Kaggle Triage Accuracy</div>
                <div class="metric-value">87.75%</div>
                <div class="metric-label">ROC-AUC: 0.9759 (OvR)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">BraTS Fusion Accuracy</div>
                <div class="metric-value">86.73%</div>
                <div class="metric-label">ROC-AUC: 0.9677</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Minority LGG Recall</div>
                <div class="metric-value">94.64%</div>
                <div class="metric-label">Surged from 0.00% Collapse</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Explainability Engine</div>
                <div class="metric-value">Dual XAI</div>
                <div class="metric-label">Grad-CAM + Attention Rollout</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Navigation Tabs
    tab_overview, tab_kaggle, tab_brats = st.tabs([
        "🏥 System Overview & Case Gallery",
        "📷 Single-Modality Diagnosis (Kaggle)",
        "🧬 Multi-Modal Fusion Grading (BraTS)",
    ])

    # ==========================================================================
    # TAB 1: SYSTEM OVERVIEW & CASE GALLERY
    # ==========================================================================
    with tab_overview:
        st.subheader("Diagnostic Workflow & Preloaded Clinical Showcase")
        st.markdown(
            """
            This platform solves two fundamental challenges in neuro-oncology machine learning:
            1. **Multi-Sequence Integration:** Stacking 4 complementary MRI contrasts (T1 native, T1-contrast, T2 water, and FLAIR edema) into an early-fusion $[4, 224, 224]$ tensor.
            2. **Algorithmic Transparency:** Using shifted-window self-attention with pixel-level Grad-CAM backpropagation and Attention Rollout to verify that model decisions align with actual pathological tumor tissue.
            """
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔍 Single-Modality 4-Class Triage")
            st.markdown(
                """
                - **Clinical Purpose:** Fast anatomical screening from routine 2D MRI scans.
                - **Classes:** High-grade parenchymal **Glioma**, extra-axial **Meningioma**, sellar **Pituitary Adenoma**, and **Healthy Brains**.
                - **Demo Available:** Preloaded cases for each tumor type available in the next tab.
                """
            )
        with c2:
            st.markdown("#### 🔬 Multi-Modal Fusion Tumor Grading")
            st.markdown(
                """
                - **Clinical Purpose:** Distinguishing indolent **Low-Grade Gliomas (LGG)** from aggressive **High-Grade Gliomas (HGG)**.
                - **Technological Breakthrough:** Resolved a severe baseline class-imbalance collapse (0% LGG recall $\\to$ 94.64% sensitivity).
                - **Visual Display:** Saliency maps overlaid directly on **FLAIR** scans to contrast edema against suppressed CSF.
                """
            )

        st.markdown("---")
        st.info("👉 **To view live inference:** Click on the **'Single-Modality Diagnosis'** or **'Multi-Modal Fusion Grading'** tabs above. Preloaded cases are already active and will generate predictions immediately!")

    # ==========================================================================
    # TAB 2: KAGGLE SINGLE-MODALITY MODE
    # ==========================================================================
    with tab_kaggle:
        st.subheader("Single-Modality 4-Class Brain Tumor Diagnosis")
        st.markdown(
            "Classifies 2D MRI scans into **Glioma**, **Meningioma**, **Pituitary**, or **No Tumor**, "
            "rendering real-time **Grad-CAM** saliency overlays."
        )

        try:
            kaggle_model, kaggle_meta = load_kaggle_model()
            k_device = kaggle_meta["device"]
            k_classes = kaggle_meta["class_names"]
        except Exception as e:
            st.error(f"❌ Failed to load Kaggle model: {e}")
            return

        # Input Mode Selector: Preloaded Demo vs Custom Upload
        col_mode, col_select = st.columns([1, 2])
        with col_mode:
            k_input_mode = st.radio(
                "Input Source:",
                ["⚡ Preloaded Clinical Demo Case", "📁 Upload Custom MRI Image"],
                key="k_mode_radio",
            )

        active_k_bytes = None
        active_k_title = ""
        active_k_desc = ""

        if k_input_mode == "⚡ Preloaded Clinical Demo Case":
            with col_select:
                k_selected_demo = st.selectbox(
                    "Select a Preloaded Clinical Case:",
                    list(KAGGLE_DEMO_SAMPLES.keys()),
                    index=0,
                )
            demo_info = KAGGLE_DEMO_SAMPLES[k_selected_demo]
            active_k_title = k_selected_demo
            active_k_desc = demo_info["description"]
            if demo_info["path"].exists():
                with open(demo_info["path"], "rb") as f:
                    active_k_bytes = f.read()
            else:
                st.warning(f"Sample file not found at {demo_info['path']}")
        else:
            with col_select:
                k_uploaded_file = st.file_uploader(
                    "Upload MRI slice (JPG / PNG):",
                    type=["jpg", "jpeg", "png"],
                    key="k_file_uploader",
                )
            if k_uploaded_file is not None:
                active_k_bytes = k_uploaded_file.getvalue()
                active_k_title = k_uploaded_file.name
                active_k_desc = "User-uploaded MRI scan."

        if active_k_bytes is None:
            st.info("👆 Please upload an image or switch to '⚡ Preloaded Clinical Demo Case' to view instant results.")
        else:
            try:
                raw_rgb, display_bg, img_tensor = preprocess_kaggle_image(active_k_bytes)
            except Exception as e:
                st.error(f"❌ Error preprocessing image: {e}")
                return

            # Run Inference
            with torch.no_grad():
                logits = kaggle_model(img_tensor.unsqueeze(0).to(k_device))
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                pred_class = k_classes[pred_idx]
                confidence = float(probs[pred_idx]) * 100.0

            st.markdown(f"**Analyzing:** `{active_k_title}` — *{active_k_desc}*")

            # Metrics and Probability Distribution
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric("Predicted Diagnostic Class", pred_class.capitalize())
                st.metric("Model Confidence", f"{confidence:.2f}%")
                if "true_label" in locals() or (k_input_mode.startswith("⚡") and "true_label" in demo_info):
                    st.success(f"Ground Truth Label: **{demo_info['true_label']}**")

            with res_col2:
                st.markdown("**Diagnostic Probability Distribution:**")
                prob_df = pd.DataFrame(
                    {
                        "Diagnostic Class": [c.capitalize() for c in k_classes],
                        "Probability (%)": probs * 100.0,
                    }
                ).set_index("Diagnostic Class")
                st.bar_chart(prob_df, y="Probability (%)")

            # Grad-CAM Explainability Section
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
                    caption=f"Original Axial MRI Slice (224x224): {active_k_title}",
                    use_container_width=True,
                )
            with col_cam:
                st.image(
                    blended_overlay,
                    caption=f"Grad-CAM Heatmap Overlay (Target: {pred_class.capitalize()} — {confidence:.2f}%)",
                    use_container_width=True,
                )

    # ==========================================================================
    # TAB 3: BRATS MULTI-MODAL FUSION MODE
    # ==========================================================================
    with tab_brats:
        st.subheader("Multi-Modal MRI Fusion Tumor Grading (BraTS 2020)")
        st.markdown(
            "Stacks co-registered **T1**, **T1ce**, **T2**, and **FLAIR** sequences into a 4-channel tensor `[4, 224, 224]` "
            "to differentiate **Low-Grade Gliomas (LGG)** from **High-Grade Gliomas (HGG)** with dual explainability."
        )

        try:
            brats_model, brats_meta = load_brats_model()
            b_device = brats_meta["device"]
            b_classes = brats_meta["class_names"]
        except Exception as e:
            st.error(f"❌ Failed to load BraTS fusion model: {e}")
            return

        # Input Mode Selector: Preloaded Demo vs Custom Upload
        b_col_mode, b_col_select = st.columns([1, 2])
        with b_col_mode:
            b_input_mode = st.radio(
                "Input Source:",
                ["⚡ Preloaded Multi-Modal Patient Case", "📁 Upload 4 Custom Sequence Scans"],
                key="b_mode_radio",
            )

        fused_tensor = None
        disp_t1 = disp_t1c = disp_t2 = disp_flair = None
        active_b_title = ""
        active_b_desc = ""
        ground_truth_grade = None

        if b_input_mode == "⚡ Preloaded Multi-Modal Patient Case":
            with b_col_select:
                b_selected_demo = st.selectbox(
                    "Select a Preloaded Patient Case:",
                    list(BRATS_DEMO_SAMPLES.keys()),
                    index=0,
                )
            demo_info = BRATS_DEMO_SAMPLES[b_selected_demo]
            active_b_title = b_selected_demo
            active_b_desc = demo_info["description"]
            ground_truth_grade = demo_info["true_grade"]

            if demo_info["path"].exists():
                data = np.load(demo_info["path"])
                t1, t1ce, t2, flair = data["t1"], data["t1ce"], data["t2"], data["flair"]
                norm_t1, disp_t1 = preprocess_brats_slice(t1)
                norm_t1c, disp_t1c = preprocess_brats_slice(t1ce)
                norm_t2, disp_t2 = preprocess_brats_slice(t2)
                norm_flair, disp_flair = preprocess_brats_slice(flair)

                fused_tensor = fuse_brats_modalities(
                    t1=norm_t1,
                    t1ce=norm_t1c,
                    t2=norm_t2,
                    flair=norm_flair,
                    target_size=(224, 224),
                    return_tensor=True,
                )
            else:
                st.warning(f"Patient file not found at {demo_info['path']}")
        else:
            with b_col_select:
                st.info("Upload all 4 co-registered modality slices (T1, T1c, T2, FLAIR) or a single multi-channel `.npz` file.")

            cu1, cu2, cu3, cu4 = st.columns(4)
            with cu1:
                f_t1 = st.file_uploader("1. T1 Native", type=["jpg", "png", "npz"], key="u_t1")
            with cu2:
                f_t1c = st.file_uploader("2. T1c Contrast", type=["jpg", "png", "npz"], key="u_t1c")
            with cu3:
                f_t2 = st.file_uploader("3. T2 Fluid", type=["jpg", "png", "npz"], key="u_t2")
            with cu4:
                f_flair = st.file_uploader("4. FLAIR", type=["jpg", "png", "npz"], key="u_flair")

            if all(f is not None for f in [f_t1, f_t1c, f_t2, f_flair]):
                try:
                    arr_t1 = decode_brats_modality_bytes(f_t1.getvalue(), f_t1.name)
                    arr_t1c = decode_brats_modality_bytes(f_t1c.getvalue(), f_t1c.name)
                    arr_t2 = decode_brats_modality_bytes(f_t2.getvalue(), f_t2.name)
                    arr_flair = decode_brats_modality_bytes(f_flair.getvalue(), f_flair.name)

                    norm_t1, disp_t1 = preprocess_brats_slice(arr_t1)
                    norm_t1c, disp_t1c = preprocess_brats_slice(arr_t1c)
                    norm_t2, disp_t2 = preprocess_brats_slice(arr_t2)
                    norm_flair, disp_flair = preprocess_brats_slice(arr_flair)

                    fused_tensor = fuse_brats_modalities(
                        t1=norm_t1,
                        t1ce=norm_t1c,
                        t2=norm_t2,
                        flair=norm_flair,
                        target_size=(224, 224),
                        return_tensor=True,
                    )
                    active_b_title = "Custom 4-Sequence Upload"
                    active_b_desc = "User-supplied multi-modal MRI sequences."
                except Exception as ex:
                    st.error(f"❌ Error processing uploaded modalities: {ex}")

        if fused_tensor is None:
            st.info("👆 Please upload all 4 modalities or switch to '⚡ Preloaded Multi-Modal Patient Case' to view instant results.")
        else:
            st.markdown(f"**Analyzing:** `{active_b_title}` — *{active_b_desc}*")

            # Display 4 Channels
            st.markdown("#### 1. Fused MRI Sequence Scans (Input Channels 0–3)")
            c_m1, c_m2, c_m3, c_m4 = st.columns(4)
            with c_m1:
                st.image(disp_t1, caption="Channel 0: T1 Native", use_container_width=True)
            with c_m2:
                st.image(disp_t1c, caption="Channel 1: T1c Post-Contrast", use_container_width=True)
            with c_m3:
                st.image(disp_t2, caption="Channel 2: T2 Fluid", use_container_width=True)
            with c_m4:
                st.image(disp_flair, caption="Channel 3: FLAIR (Edema)", use_container_width=True)

            # Inference
            with torch.no_grad():
                logits = brats_model(fused_tensor.unsqueeze(0).to(b_device))
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                pred_grade = b_classes[pred_idx]
                confidence = float(probs[pred_idx]) * 100.0

            grade_title = "High-Grade Glioma (HGG)" if pred_grade == "HGG" else "Low-Grade Glioma (LGG)"

            # Metrics
            st.markdown("---")
            st.markdown("#### 2. Diagnostic Grade Prediction")
            b_res1, b_res2 = st.columns([1, 2])
            with b_res1:
                st.metric("Predicted Tumor Grade", grade_title)
                st.metric("Model Confidence", f"{confidence:.2f}%")
                if ground_truth_grade:
                    st.success(f"Ground Truth Grade: **{ground_truth_grade}**")

            with b_res2:
                st.markdown("**Grade Probability Breakdown:**")
                prob_b_df = pd.DataFrame(
                    {
                        "Tumor Grade": ["Low-Grade Glioma (LGG)", "High-Grade Glioma (HGG)"],
                        "Probability (%)": probs * 100.0,
                    }
                ).set_index("Tumor Grade")
                st.bar_chart(prob_b_df, y="Probability (%)")

            # Dual Explainability
            st.markdown("---")
            st.markdown("#### 3. Dual Visual Explainability (Overlaid on FLAIR Modality)")
            st.markdown(
                "FLAIR suppresses cerebrospinal fluid to highlight peritumoral vasogenic edema. "
                "Below are the **Grad-CAM** gradient saliency overlay and the **Swin Attention Rollout** self-attention map."
            )

            with st.spinner("Computing Grad-CAM and Swin Attention Rollout heatmaps..."):
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

            exp_c1, exp_c2, exp_c3 = st.columns(3)
            with exp_c1:
                st.image(disp_flair, caption="Representative FLAIR Modality", use_container_width=True)
            with exp_c2:
                st.image(g_blended, caption=f"Grad-CAM Saliency ({pred_grade} — {confidence:.1f}%)", use_container_width=True)
            with exp_c3:
                st.image(r_blended, caption="Swin Attention Rollout (Attention Flow)", use_container_width=True)

    # Footer
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
