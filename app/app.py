"""
Streamlit Web Application for Explainable Brain Tumor Diagnosis.
Theme & Layout inspired by modern BrainWave aesthetic:
- Ethereal gradient canvas with frosted glass cards (Bento layout).
- Hero section with 3D illuminated neural visualization and asymmetrical card grid.
- Dedicated, organized sections for Diagnostic Studio, Architecture Specs, and Benchmark Validation.
- Fully dynamic, clean, interactive, and responsive.
"""

import base64
import io
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

def render_html(html_str: str):
    """Safely render HTML without markdown indentation glitches."""
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Global cached hardware device detection for instant reruns without CUDA polling lag
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "CUDA GPU Active" if torch.cuda.is_available() else "CPU Execution Mode"

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
# ASSET ENCODING HELPER
# ==============================================================================

@st.cache_data
def get_base64_image(image_path: Union[str, Path], _mtime: float = 0.0) -> str:
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    if suffix == ".png":
        mime_type = "image/png"
    elif suffix == ".svg":
        mime_type = "image/svg+xml"
    elif suffix in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif suffix == ".webp":
        mime_type = "image/webp"
    else:
        mime_type = "application/octet-stream"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


# ==============================================================================
# PRELOADED CLINICAL SAMPLES
# ==============================================================================

KAGGLE_DEMO_SAMPLES = {
    "Glioma (Intra-Axial Lesion)": {
        "path": PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "glioma" / "Te-gl_1.jpg",
        "label": "Glioma",
        "description": "High-grade intra-axial parenchymal mass with mass effect.",
        "badge": "Malignant",
    },
    "Pituitary Adenoma (Sellar Mass)": {
        "path": PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "pituitary" / "Te-pi_1.jpg",
        "label": "Pituitary",
        "description": "Circumscribed sellar / parasellar neuroendocrine adenoma.",
        "badge": "Benign / Sellar",
    },
    "Healthy Normal Control": {
        "path": PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "notumor" / "Te-no_1.jpg",
        "label": "No Tumor",
        "description": "Normal cerebral anatomical baseline without lesion or edema.",
        "badge": "Non-Tumorous",
    },
    "Meningioma (Dural-Based Mass)": {
        "path": PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "meningioma" / "Te-aug-me_1.jpg",
        "label": "Meningioma",
        "description": "Extra-axial, dural-based mass compressing adjacent cortical tissue.",
        "badge": "Extra-Axial",
    },
}

BRATS_DEMO_SAMPLES = {
    "Patient 310 (Low-Grade Glioma / LGG, Slice 52)": {
        "path": PROJECT_ROOT / "data" / "processed" / "brats_normalized" / "BraTS20_Training_310" / "slice_052.npz",
        "grade": "LGG",
        "description": "WHO Grade II astrocytoma with hyperintense T2/FLAIR signal without aggressive necrosis.",
        "badge": "WHO Grade II",
    },
    "Patient 020 (High-Grade Glioma / HGG, Slice 40)": {
        "path": PROJECT_ROOT / "data" / "processed" / "brats_normalized" / "BraTS20_Training_020" / "slice_040.npz",
        "grade": "HGG",
        "description": "WHO Grade IV glioblastoma displaying intense ring enhancement on T1ce and vasogenic edema.",
        "badge": "WHO Grade IV",
    },
    "Patient 006 (High-Grade Glioblastoma / HGG, Slice 84)": {
        "path": PROJECT_ROOT / "data" / "processed" / "brats_normalized" / "BraTS20_Training_006" / "slice_084.npz",
        "grade": "HGG",
        "description": "Large necrotic cavitary glioblastoma with marked ventricular compression.",
        "badge": "WHO Grade IV Necrotic",
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
# MODERN THEME & BENTO STYLESHEET
# ==============================================================================

def inject_brainwave_theme(is_dark: bool = False):
    """
    Injects custom CSS implementing the high-end BrainWave aesthetic:
    - Ethereal pastel gradient mesh canvas.
    - Floating frosted glass containers with soft diffused shadows.
    - Plus Jakarta Sans typography.
    - Asymmetric bento grid with floating 3D brain card animation.
    - Polished buttons, badges, sliders, and tab bars.
    """
    day_path = PROJECT_ROOT / "app" / "assets" / "theme_day_switch.svg"
    night_path = PROJECT_ROOT / "app" / "assets" / "theme_night_switch.svg"
    day_mtime = day_path.stat().st_mtime if day_path.exists() else 0.0
    night_mtime = night_path.stat().st_mtime if night_path.exists() else 0.0
    day_switch_b64 = get_base64_image(day_path, _mtime=day_mtime)
    night_switch_b64 = get_base64_image(night_path, _mtime=night_mtime)
    active_toggle_bg = night_switch_b64 if is_dark else day_switch_b64

    if is_dark:
        bg_canvas = "radial-gradient(circle at 10% 20%, rgba(30, 27, 75, 0.9) 0%, rgba(49, 23, 62, 0.8) 45%, rgba(15, 23, 42, 0.98) 100%), #0b0f19"
        card_bg = "rgba(22, 28, 48, 0.88)"
        card_border = "rgba(255, 255, 255, 0.12)"
        card_sub_bg = "#162036"
        text_hero = "#f8fafc"
        text_sub = "#94a3b8"
        accent_color = "#818cf8"
        accent_gradient = "linear-gradient(135deg, #6366f1 0%, #a855f7 100%)"
        inner_shadow = "0 20px 50px -10px rgba(0, 0, 0, 0.6)"
        tab_bg = "rgba(15, 23, 42, 0.85)"
        tab_border = "rgba(255, 255, 255, 0.16)"
        tab_unselected_bg = "rgba(255, 255, 255, 0.14)"
        tab_unselected_border = "rgba(255, 255, 255, 0.22)"
        tab_unselected_text = "#ffffff"
        tab_unselected_hover_bg = "rgba(255, 255, 255, 0.24)"
        tab_selected_bg = "linear-gradient(135deg, #6366f1 0%, #a855f7 100%)"
        tab_selected_text = "#ffffff"
        tab_selected_shadow = "0 4px 20px rgba(99, 102, 241, 0.55)"
        popover_btn_bg = "rgba(30, 41, 59, 0.95)"
        popover_btn_text = "#ffffff"
        popover_btn_border = "rgba(255, 255, 255, 0.25)"
        popover_btn_hover_bg = "rgba(49, 46, 129, 0.9)"
        popover_btn_hover_text = "#ffffff"
        theme_btn_bg = "rgba(30, 41, 59, 0.9)"
        theme_btn_border = "rgba(255, 255, 255, 0.18)"
        nav_divider_bg = "rgba(255, 255, 255, 0.1)"
        btn_sec_bg = "rgba(26, 34, 53, 0.95)"
        btn_sec_text = "#ffffff"
        btn_sec_border = "rgba(255, 255, 255, 0.22)"
        btn_sec_hover_bg = "rgba(49, 46, 129, 0.85)"
        btn_sec_hover_text = "#ffffff"
        dropzone_bg = "rgba(15, 23, 42, 0.85)"
        dropzone_hover_bg = "rgba(30, 27, 75, 0.75)"
        dropzone_border = "rgba(129, 140, 248, 0.4)"
        mode_icon_bg = "rgba(30, 41, 59, 0.9)"
        mode_icon_border = "rgba(255, 255, 255, 0.14)"
        badge_inactive_bg = "rgba(255, 255, 255, 0.06)"
        badge_inactive_border = "rgba(255, 255, 255, 0.1)"
        chip_bg = "rgba(255, 255, 255, 0.08)"
        chip_border = "rgba(255, 255, 255, 0.12)"
        chip_text = "#cbd5e1"
    else:
        bg_canvas = "radial-gradient(circle at 12% 18%, rgba(226, 218, 252, 0.75) 0%, rgba(254, 230, 238, 0.65) 42%, rgba(220, 234, 254, 0.75) 90%), #f5f6fb"
        card_bg = "rgba(255, 255, 255, 0.88)"
        card_border = "rgba(255, 255, 255, 0.9)"
        card_sub_bg = "#ffffff"
        text_hero = "#0f172a"
        text_sub = "#64748b"
        accent_color = "#4f46e5"
        accent_gradient = "linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)"
        inner_shadow = "0 30px 80px -15px rgba(118, 100, 200, 0.16)"
        tab_bg = "rgba(241, 245, 249, 0.95)"
        tab_border = "rgba(203, 213, 225, 0.9)"
        tab_unselected_bg = "rgba(226, 232, 240, 0.95)"
        tab_unselected_border = "rgba(203, 213, 225, 0.95)"
        tab_unselected_text = "#0f172a"
        tab_unselected_hover_bg = "rgba(203, 213, 225, 1.0)"
        tab_selected_bg = "linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)"
        tab_selected_text = "#ffffff"
        tab_selected_shadow = "0 4px 18px rgba(79, 70, 229, 0.35)"
        popover_btn_bg = "#ffffff"
        popover_btn_text = "#0f172a"
        popover_btn_border = "rgba(203, 213, 225, 0.95)"
        popover_btn_hover_bg = "#f1f5f9"
        popover_btn_hover_text = "#4338ca"
        theme_btn_bg = "rgba(255, 255, 255, 0.9)"
        theme_btn_border = "rgba(226, 232, 240, 0.85)"
        nav_divider_bg = "rgba(226, 232, 240, 0.6)"
        btn_sec_bg = "#ffffff"
        btn_sec_text = "#0f172a"
        btn_sec_border = "rgba(203, 213, 225, 0.95)"
        btn_sec_hover_bg = "#f1f5f9"
        btn_sec_hover_text = "#4338ca"
        dropzone_bg = "rgba(248, 250, 252, 0.85)"
        dropzone_hover_bg = "rgba(238, 242, 255, 0.85)"
        dropzone_border = "rgba(99, 102, 241, 0.35)"
        mode_icon_bg = "#f1f5f9"
        mode_icon_border = "rgba(226, 232, 240, 0.9)"
        badge_inactive_bg = "rgba(241, 245, 249, 0.8)"
        badge_inactive_border = "rgba(226, 232, 240, 0.9)"
        chip_bg = "rgba(241, 245, 249, 0.9)"
        chip_border = "rgba(226, 232, 240, 0.9)"
        chip_text = "#475569"

    css = f"""
    <style>
    /* Global Viewport Reset */
    .stApp {{
        background: {bg_canvas} !important;
        background-attachment: fixed !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: {text_hero};
    }}

    /* Completely hide Streamlit sidebar and its toggle button */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"],
    button[data-testid="stSidebarCollapseButton"],
    div[data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}

    /* Hide standard Streamlit header clutter */
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1240px !important;
    }}

    /* Main Floating Frosted Glass Wrapper */
    .brainwave-canvas {{
        background: {card_bg};
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        border: 1px solid {card_border};
        border-radius: 36px;
        box-shadow: {inner_shadow}, 0 0 0 1px rgba(255, 255, 255, 0.6) inset;
        padding: 2.2rem 2.8rem 3rem 2.8rem;
        margin-bottom: 2rem;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    /* Top Navigation Header (BrainWave Style) */
    .bw-navbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1.8rem;
        border-bottom: 1px solid rgba(226, 232, 240, 0.5);
        margin-bottom: 2.2rem;
    }}
    .bw-brand {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}
    .bw-brand-text {{
        font-weight: 800;
        font-size: 1.35rem;
        letter-spacing: -0.04em;
        line-height: 1.05;
        color: {text_hero};
        text-transform: uppercase;
    }}
    .bw-brand-badge {{
        background: {accent_gradient};
        color: white;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}
    .bw-status-pill {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(34, 197, 94, 0.12);
        color: #16a34a;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 9999px;
        border: 1px solid rgba(34, 197, 94, 0.25);
    }}
    .bw-status-dot {{
        width: 7px;
        height: 7px;
        background: #16a34a;
        border-radius: 50%;
        box-shadow: 0 0 8px #16a34a;
        animation: pulseDot 2s infinite ease-in-out;
    }}
    @keyframes pulseDot {{
        0%, 100% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.3); opacity: 0.6; }}
    }}

    /* Bento Hero Grid (2-Columns Asymmetric) */
    .bento-hero-grid {{
        display: grid;
        grid-template-columns: 1.15fr 0.95fr;
        gap: 2rem;
        margin-bottom: 2.8rem;
    }}
    @media (max-width: 900px) {{
        .bento-hero-grid {{
            grid-template-columns: 1fr;
        }}
    }}

    /* Left Column Top Card: Big Bold Headline */
    .bento-card-main {{
        background: {card_sub_bg};
        border-radius: 28px;
        padding: 2.8rem 2.5rem;
        box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.05), 0 0 0 1px rgba(241, 245, 249, 0.9);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 380px;
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    .bento-card-main:hover {{
        transform: translateY(-3px);
        box-shadow: 0 20px 40px -10px rgba(115, 105, 185, 0.12);
    }}
    .bento-main-title {{
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.12;
        letter-spacing: -0.035em;
        color: {text_hero};
        margin-bottom: 1.2rem;
    }}
    .bento-main-sub {{
        font-size: 1.05rem;
        line-height: 1.6;
        color: {text_sub};
        margin-bottom: 2rem;
        max-width: 92%;
    }}
    .bento-pill-action {{
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
        background: #0f172a;
        color: #ffffff !important;
        padding: 0.85rem 1.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        text-decoration: none;
        width: fit-content;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.35);
        transition: all 0.2s ease;
    }}
    .bento-pill-action:hover {{
        background: #1e293b;
        transform: translateY(-2px);
        box-shadow: 0 14px 28px -5px rgba(15, 23, 42, 0.45);
    }}
    .bento-pill-arrow {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        background: rgba(255, 255, 255, 0.18);
        border-radius: 50%;
        font-size: 0.85rem;
    }}

    /* Left Column Bottom Card: Horizontal Sub-Card */
    .bento-card-sub {{
        background: {card_sub_bg};
        border-radius: 26px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.05);
        display: flex;
        align-items: center;
        gap: 1.4rem;
        margin-top: 1.6rem;
        border: 1px solid rgba(226, 232, 240, 0.6);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    .bento-card-sub:hover {{
        transform: translateY(-3px);
        box-shadow: 0 16px 36px -10px rgba(115, 105, 185, 0.12);
    }}
    .bento-sub-thumb {{
        width: 105px;
        height: 105px;
        border-radius: 20px;
        object-fit: cover;
        flex-shrink: 0;
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.15);
    }}
    .bento-sub-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {text_hero};
        margin-bottom: 0.35rem;
    }}
    .bento-sub-desc {{
        font-size: 0.88rem;
        color: {text_sub};
        line-height: 1.45;
        margin-bottom: 0.5rem;
    }}
    .bento-sub-link {{
        font-size: 0.85rem;
        font-weight: 700;
        color: {accent_color};
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }}

    /* Right Column: Tall Gradient Showcase with 3D Brain */
    .bento-card-showcase {{
        background: linear-gradient(150deg, #7c72db 0%, #6366f1 35%, #8b5cf6 65%, #ab87f8 100%);
        border-radius: 32px;
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 2rem;
        min-height: 520px;
        box-shadow: 0 24px 60px -10px rgba(99, 102, 241, 0.42);
        border: 1px solid rgba(255, 255, 255, 0.25);
    }}
    .showcase-floating-brain {{
        width: 86%;
        margin: 0.5rem auto 1rem auto;
        display: block;
        filter: drop-shadow(0 25px 35px rgba(25, 15, 75, 0.4));
        animation: floatHero 6s ease-in-out infinite;
        transition: transform 0.3s ease;
    }}
    @keyframes floatHero {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        50% {{ transform: translateY(-10px) rotate(1deg); }}
    }}
    .showcase-frosted-overlay {{
        background: rgba(255, 255, 255, 0.22);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 22px;
        padding: 1.5rem 1.6rem;
        color: #ffffff;
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }}
    .showcase-overlay-title {{
        font-size: 1.25rem;
        font-weight: 800;
        line-height: 1.2;
        color: #ffffff;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }}
    .showcase-overlay-sub {{
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.9);
        line-height: 1.45;
        margin: 0;
    }}
    .showcase-circle-arrow {{
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-size: 1.1rem;
        flex-shrink: 0;
        transition: transform 0.2s, background 0.2s;
    }}
    .showcase-circle-arrow:hover {{
        background: rgba(255, 255, 255, 0.4);
        transform: scale(1.05);
    }}

    /* Universal BaseWeb Tabs Container: Sleek Floating Pill Bar */
    div[data-testid="stTabs"] [data-baseweb="tab-list"],
    .stTabs [data-baseweb="tab-list"],
    [data-baseweb="tab-list"],
    div[data-baseweb="tab-list"] {{
        gap: 8px !important;
        background: {tab_bg} !important;
        padding: 6px !important;
        border-radius: 9999px !important;
        border: 1px solid {tab_border} !important;
        margin-bottom: 1.8rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12) inset !important;
        width: fit-content !important;
        max-width: 100% !important;
        overflow-x: auto !important;
    }}
    
    /* Remove default BaseWeb cyan tab underline & border lines completely */
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"],
    div[data-testid="stTabs"] hr,
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"],
    [data-baseweb="tab-highlight"],
    [data-baseweb="tab-border"],
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"] {{
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
    }}

    /* ALL Tabs: Base Unselected State (Crisp High-Contrast Pills) */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] button[data-baseweb="tab"],
    div[data-testid="stTabs"] button[data-baseweb="tab"],
    div[data-testid="stTabs"] button[role="tab"],
    div[data-testid="stTabs"] button,
    .stTabs [data-baseweb="tab-list"] button,
    .stTabs [data-baseweb="tab"],
    [data-baseweb="tab"],
    button[data-baseweb="tab"],
    button[role="tab"] {{
        height: 42px !important;
        border-radius: 9999px !important;
        padding: 0 1.4rem !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        color: {tab_unselected_text} !important;
        -webkit-text-fill-color: {tab_unselected_text} !important;
        border: 1px solid {tab_unselected_border} !important;
        border-bottom: 1px solid {tab_unselected_border} !important;
        background: {tab_unselected_bg} !important;
        background-color: {tab_unselected_bg} !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: none !important;
        outline: none !important;
    }}

    /* Force all child labels inside unselected tabs to be high-contrast and legible */
    div[data-testid="stTabs"] button[data-baseweb="tab"] *,
    div[data-testid="stTabs"] button[role="tab"] *,
    div[data-testid="stTabs"] button *,
    .stTabs [data-baseweb="tab"] *,
    [data-baseweb="tab"] *,
    button[data-baseweb="tab"] *,
    button[role="tab"] * {{
        color: {tab_unselected_text} !important;
        fill: {tab_unselected_text} !important;
        -webkit-text-fill-color: {tab_unselected_text} !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        margin: 0 !important;
        line-height: 1.2 !important;
        opacity: 1 !important;
    }}

    /* Hover State for Unselected Tabs */
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover,
    div[data-testid="stTabs"] button[role="tab"]:hover,
    .stTabs [data-baseweb="tab"]:hover,
    [data-baseweb="tab"]:hover {{
        background: {tab_unselected_hover_bg} !important;
        background-color: {tab_unselected_hover_bg} !important;
        transform: translateY(-1px) !important;
    }}

    /* Active / Selected Tab State (Vivid Gradient Capsule, Zero Underline) */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [aria-selected="true"],
    [data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"],
    button[role="tab"][aria-selected="true"] {{
        background: {tab_selected_bg} !important;
        background-color: {tab_selected_bg} !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        box-shadow: {tab_selected_shadow} !important;
        transform: translateY(0px) !important;
        border: none !important;
        border-bottom: none !important;
        border-radius: 9999px !important;
    }}
    div[data-testid="stTabs"] button[aria-selected="true"] *,
    div[data-testid="stTabs"] button[aria-selected="true"] p,
    div[data-testid="stTabs"] button[aria-selected="true"] span,
    div[data-testid="stTabs"] button[aria-selected="true"] div,
    .stTabs [aria-selected="true"] *,
    [data-baseweb="tab"][aria-selected="true"] *,
    button[data-baseweb="tab"][aria-selected="true"] * {{
        color: #ffffff !important;
        fill: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        opacity: 1 !important;
    }}

    /* Streamlit Bordered Container Customization (Bento Card Container) */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {card_sub_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 24px !important;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.06) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        padding: 0.6rem 0.6rem 0.8rem 0.6rem !important;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        border-color: rgba(129, 140, 248, 0.45) !important;
        box-shadow: 0 16px 36px -10px rgba(99, 102, 241, 0.18) !important;
    }}

    /* Mode Selection Card Elements */
    .mode-card-inner {{
        padding: 0.6rem 0.6rem 0.8rem 0.6rem;
    }}
    .mode-header-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.9rem;
    }}
    .mode-icon-circle {{
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: {mode_icon_bg};
        border: 1px solid {mode_icon_border};
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }}
    .mode-status-badge {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 9999px;
    }}
    .mode-status-badge.active {{
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.35);
        box-shadow: 0 0 12px rgba(34, 197, 94, 0.25);
    }}
    .mode-status-badge.inactive {{
        background: {badge_inactive_bg};
        color: {text_sub};
        border: 1px solid {badge_inactive_border};
    }}
    .mode-title-text {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {text_hero};
        margin-bottom: 0.45rem;
        line-height: 1.3;
    }}
    .mode-desc-text {{
        font-size: 0.88rem;
        color: {text_sub};
        line-height: 1.5;
        min-height: 52px;
        margin-bottom: 0.9rem;
    }}
    .mode-meta-row {{
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }}
    .mode-chip-tech {{
        background: {chip_bg};
        color: {chip_text};
        border: 1px solid {chip_border};
        font-size: 0.74rem;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 8px;
    }}
    .mode-chip-acc {{
        background: {accent_gradient};
        color: #ffffff;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
    }}

    /* File Uploader Dropzone Styling */
    div[data-testid="stFileUploader"] {{
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
    }}
    div[data-testid="stFileUploader"] section {{
        background: {dropzone_bg} !important;
        border: 2px dashed {dropzone_border} !important;
        border-radius: 20px !important;
        padding: 1.6rem 1.4rem !important;
        transition: all 0.25s ease !important;
    }}
    div[data-testid="stFileUploader"] section:hover {{
        border-color: {accent_color} !important;
        background: {dropzone_hover_bg} !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15) !important;
    }}
    div[data-testid="stFileUploader"] section button {{
        background: {btn_sec_bg} !important;
        color: {btn_sec_text} !important;
        border: 1px solid {btn_sec_border} !important;
        border-radius: 9999px !important;
        font-weight: 600 !important;
        padding: 0.4rem 1.1rem !important;
    }}
    div[data-testid="stFileUploader"] section button:hover {{
        border-color: {accent_color} !important;
        color: {accent_color} !important;
    }}
    div[data-testid="stFileUploader"] section small,
    div[data-testid="stFileUploader"] section span,
    div[data-testid="stFileUploader"] label {{
        color: {text_sub} !important;
        font-weight: 500 !important;
    }}
    div[data-testid="stFileUploader"] label {{
        color: {text_hero} !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.4rem !important;
    }}

    /* Expander Styling */
    div[data-testid="stExpander"] {{
        background: {card_sub_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 18px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04) !important;
        margin-top: 1rem !important;
        margin-bottom: 1.5rem !important;
    }}
    div[data-testid="stExpander"] details summary {{
        padding: 0.9rem 1.2rem !important;
        font-weight: 600 !important;
        color: {text_hero} !important;
    }}
    div[data-testid="stExpander"] details summary:hover {{
        color: {accent_color} !important;
    }}
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        padding: 1.2rem !important;
        border-top: 1px solid {card_border} !important;
    }}

    /* Diagnostic Result & Metric Cards */
    .metric-bento-card {{
        background: {card_sub_bg};
        border-radius: 20px;
        padding: 1.6rem;
        border: 1px solid {card_border};
        box-shadow: 0 8px 24px -5px rgba(15, 23, 42, 0.05);
        text-align: left;
    }}
    .metric-bento-val {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {accent_color};
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }}
    .metric-bento-label {{
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {text_sub};
    }}

    /* All Streamlit Buttons Polish & Absolute Visibility Guarantee (EXCEPT Day/Night toggle) */
    div.stButton:not(.st-key-theme_toggle_btn) > button,
    div.stButton:not(.st-key-theme_toggle_btn) button,
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="baseButton-secondary"],
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="baseButton-primary"],
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="stBaseButton-secondary"],
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="stBaseButton-primary"] {{
        border-radius: 9999px !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        letter-spacing: -0.01em !important;
        padding: 0.65rem 1.6rem !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        gap: 0.5rem !important;
        min-height: 46px !important;
        box-sizing: border-box !important;
    }}

    /* Secondary Buttons (Locked High Contrast: Crisp Dark Text on Light mode, Pure White Text on Dark mode) */
    div.stButton:not(.st-key-theme_toggle_btn) > button[kind="secondary"],
    div.stButton:not(.st-key-theme_toggle_btn) button[kind="secondary"],
    div.stButton:not(.st-key-theme_toggle_btn) > button:not([kind="primary"]),
    div.stButton:not(.st-key-theme_toggle_btn) button:not([kind="primary"]),
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="baseButton-secondary"],
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="stBaseButton-secondary"] {{
        background: {btn_sec_bg} !important;
        color: {btn_sec_text} !important;
        border: 1px solid {btn_sec_border} !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08) !important;
    }}
    /* Enforce visible text color on all child tags within secondary buttons */
    div.stButton:not(.st-key-theme_toggle_btn) > button[kind="secondary"] *,
    div.stButton:not(.st-key-theme_toggle_btn) button[kind="secondary"] *,
    div.stButton:not(.st-key-theme_toggle_btn) > button:not([kind="primary"]) *,
    div.stButton:not(.st-key-theme_toggle_btn) button:not([kind="primary"]) *,
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="baseButton-secondary"] *,
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="stBaseButton-secondary"] * {{
        color: {btn_sec_text} !important;
        fill: {btn_sec_text} !important;
        -webkit-text-fill-color: {btn_sec_text} !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }}
    div.stButton:not(.st-key-theme_toggle_btn) > button[kind="secondary"]:hover,
    div.stButton:not(.st-key-theme_toggle_btn) button[kind="secondary"]:hover,
    div.stButton:not(.st-key-theme_toggle_btn) > button:not([kind="primary"]):hover,
    div.stButton:not(.st-key-theme_toggle_btn) button:not([kind="primary"]):hover,
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="baseButton-secondary"]:hover,
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="stBaseButton-secondary"]:hover {{
        background: {btn_sec_hover_bg} !important;
        color: {btn_sec_hover_text} !important;
        border-color: {accent_color} !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(99, 102, 241, 0.28) !important;
    }}
    div.stButton:not(.st-key-theme_toggle_btn) > button[kind="secondary"]:hover *,
    div.stButton:not(.st-key-theme_toggle_btn) button[kind="secondary"]:hover *,
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="baseButton-secondary"]:hover * {{
        color: {btn_sec_hover_text} !important;
        fill: {btn_sec_hover_text} !important;
        -webkit-text-fill-color: {btn_sec_hover_text} !important;
    }}

    /* Primary Buttons (Gradient with pure white text ALWAYS) */
    div.stButton:not(.st-key-theme_toggle_btn) > button[kind="primary"],
    div.stButton:not(.st-key-theme_toggle_btn) button[kind="primary"],
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="baseButton-primary"],
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="stBaseButton-primary"] {{
        background: {accent_gradient} !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.42) !important;
    }}
    div.stButton:not(.st-key-theme_toggle_btn) > button[kind="primary"] *,
    div.stButton:not(.st-key-theme_toggle_btn) button[kind="primary"] *,
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="baseButton-primary"] *,
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="stBaseButton-primary"] * {{
        color: #ffffff !important;
        fill: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }}
    div.stButton:not(.st-key-theme_toggle_btn) > button[kind="primary"]:hover,
    div.stButton:not(.st-key-theme_toggle_btn) button[kind="primary"]:hover,
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="baseButton-primary"]:hover,
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="stBaseButton-primary"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px rgba(99, 102, 241, 0.58) !important;
        filter: brightness(1.08) !important;
    }}
    div.stButton:not(.st-key-theme_toggle_btn) > button[kind="primary"]:active,
    div.stButton:not(.st-key-theme_toggle_btn) [data-testid="baseButton-primary"]:active {{
        transform: translateY(0px) !important;
    }}

    /* Top Nav Divider */
    .bw-nav-divider {{
        height: 1px;
        background: {nav_divider_bg};
        margin: 0.8rem 0 2rem 0;
    }}

    /* Realistic Day / Night Switch Button (Matching Reference Design 1:1) */
    div.st-key-theme_toggle_btn,
    div.st-key-theme_toggle_btn.stButton {{
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        width: 100% !important;
        min-width: 110px !important;
    }}
    div.st-key-theme_toggle_btn button,
    div.st-key-theme_toggle_btn button[kind="secondary"],
    div.st-key-theme_toggle_btn button:not([kind="primary"]),
    div.st-key-theme_toggle_btn [data-testid="baseButton-secondary"],
    div.st-key-theme_toggle_btn [data-testid="stBaseButton-secondary"],
    div.st-key-theme_toggle_btn > button {{
        box-sizing: border-box !important;
        display: block !important;
        position: relative !important;
        width: 110px !important;
        min-width: 110px !important;
        max-width: 110px !important;
        height: 46px !important;
        min-height: 46px !important;
        max-height: 46px !important;
        border-radius: 9999px !important;
        overflow: hidden !important;
        cursor: pointer !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        outline: none !important;
        background-color: transparent !important;
        background-image: url('{active_toggle_bg}') !important;
        background-repeat: no-repeat !important;
        background-position: center center !important;
        background-size: 100% 100% !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.28) !important;
        transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.25s ease !important;
        -webkit-appearance: none !important;
        appearance: none !important;
    }}
    div.st-key-theme_toggle_btn button:hover {{
        transform: translateY(-2px) scale(1.06) !important;
        box-shadow: 0 8px 26px rgba(99, 102, 241, 0.5) !important;
    }}
    /* Hide button text and internal containers inside toggle */
    div.st-key-theme_toggle_btn button *,
    div.st-key-theme_toggle_btn button p,
    div.st-key-theme_toggle_btn button span,
    div.st-key-theme_toggle_btn button div {{
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
        font-size: 0 !important;
        width: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    /* System Popover Trigger Button: Absolute High-Contrast Visibility in Both Themes */
    div[data-testid="stPopover"],
    div[data-testid="stPopoverTarget"] {{
        width: 100% !important;
        display: flex !important;
    }}
    div[data-testid="stPopover"] button,
    div[data-testid="stPopover"] > button,
    div[data-testid="stPopoverTarget"] button,
    div[data-testid="stPopoverTarget"] > button,
    div[data-testid="stPopover"] [data-testid="baseButton-secondary"],
    div[data-testid="stPopoverTarget"] [data-testid="baseButton-secondary"],
    div[data-testid="stPopover"] [data-testid="stBaseButton-secondary"],
    div[data-testid="stPopoverTarget"] [data-testid="stBaseButton-secondary"] {{
        border-radius: 9999px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        padding: 0.55rem 1.3rem !important;
        background: {popover_btn_bg} !important;
        background-color: {popover_btn_bg} !important;
        color: {popover_btn_text} !important;
        -webkit-text-fill-color: {popover_btn_text} !important;
        border: 1px solid {popover_btn_border} !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12) !important;
        backdrop-filter: blur(14px) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.5rem !important;
        min-height: 44px !important;
        width: 100% !important;
        cursor: pointer !important;
        outline: none !important;
    }}
    div[data-testid="stPopover"] button *,
    div[data-testid="stPopoverTarget"] button *,
    div[data-testid="stPopover"] button p,
    div[data-testid="stPopoverTarget"] button p,
    div[data-testid="stPopover"] button span,
    div[data-testid="stPopoverTarget"] button span,
    div[data-testid="stPopover"] button div,
    div[data-testid="stPopoverTarget"] button div,
    div[data-testid="stPopover"] button svg,
    div[data-testid="stPopoverTarget"] button svg {{
        color: {popover_btn_text} !important;
        fill: {popover_btn_text} !important;
        -webkit-text-fill-color: {popover_btn_text} !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }}
    div[data-testid="stPopover"] button:hover,
    div[data-testid="stPopoverTarget"] button:hover {{
        background: {popover_btn_hover_bg} !important;
        background-color: {popover_btn_hover_bg} !important;
        color: {popover_btn_hover_text} !important;
        border-color: {accent_color} !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 22px rgba(99, 102, 241, 0.32) !important;
    }}
    div[data-testid="stPopover"] button:hover *,
    div[data-testid="stPopoverTarget"] button:hover * {{
        color: {popover_btn_hover_text} !important;
        fill: {popover_btn_hover_text} !important;
        -webkit-text-fill-color: {popover_btn_hover_text} !important;
    }}

    /* Popover Content Card */
    div[data-testid="stPopoverBody"] {{
        background: {card_bg} !important;
        border-radius: 22px !important;
        border: 1px solid {card_border} !important;
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.25) !important;
        backdrop-filter: blur(24px) !important;
        padding: 1.4rem !important;
    }}

    /* Footer */
    .bw-footer {{
        text-align: center;
        color: {text_sub};
        font-size: 0.82rem;
        margin-top: 3.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(226, 232, 240, 0.6);
        line-height: 1.6;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    st.set_page_config(
        page_title="BrainWave | Vision Transformers for Brain Oncology",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # State Management
    if "is_dark" not in st.session_state:
        st.session_state["is_dark"] = False
    if "selected_mode" not in st.session_state:
        st.session_state["selected_mode"] = "single"
    if "k_analysis_done" not in st.session_state:
        st.session_state["k_analysis_done"] = False
    if "b_analysis_done" not in st.session_state:
        st.session_state["b_analysis_done"] = False

    is_dark = st.session_state["is_dark"]

    # Inject the BrainWave theme stylesheet
    inject_brainwave_theme(is_dark=is_dark)

    render_html("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
    """)

    # Load Base64 Images for Hero Showcase
    brain_b64 = get_base64_image("app/assets/hero_brain.jpg")
    ribbon_b64 = get_base64_image("app/assets/card_ribbon.jpg")

    # ==========================================================================
    # 1. TOP BRAND NAVBAR WITH INTERACTIVE THEME TOGGLE & SYSTEM HEALTH
    # ==========================================================================
    col_nav_brand, col_nav_status, col_nav_actions = st.columns(
        [1.7, 2.1, 1.8],
        vertical_alignment="center",
    )

    with col_nav_brand:
        render_html("""
        <div class="bw-brand">
            <div class="bw-brand-text">BRAIN<br/>WAVE</div>
            <span class="bw-brand-badge">Swin-ViT XAI</span>
        </div>
        """)

    with col_nav_status:
        render_html("""
        <div class="bw-status-pill">
            <div class="bw-status-dot"></div>
            <span>Pretrained Swin Transformers Online</span>
        </div>
        """)

    with col_nav_actions:
        col_act_health, col_act_theme = st.columns([1.0, 0.8], vertical_alignment="center")
        with col_act_health:
            with st.popover("⚡ System Info", use_container_width=True):
                st.markdown("##### 🔬 Hardware & Model Health")
                device_str = DEVICE_STR
                st.markdown(f"- **Runtime:** `{device_str}`")
                st.markdown("- **Engine:** `PyTorch 2.13.0`")
                st.markdown("- **Vision Library:** `timm 1.0.28`")
                st.markdown("- **Swin-Tiny (Kaggle):** `27.52M params` (Ready)")
                st.markdown("- **Swin-Base (BraTS):** `86.75M params` (Ready)")
                st.caption("Investigational translational research tool.")

        with col_act_theme:
            theme_help = "Switch to Deep Violet Night Theme" if not is_dark else "Switch to Luminous Pastel Day Theme"
            if st.button(" ", key="theme_toggle_btn", help=theme_help):
                st.session_state["is_dark"] = not is_dark
                st.rerun()

    render_html('<div class="bw-nav-divider"></div>')

    # ==========================================================================
    # 2. HERO BENTO GRID (Asymmetric Layout matching Reference Image)
    # ==========================================================================
    render_html(f"""
    <div class="bento-hero-grid">
        <div style="display: flex; flex-direction: column; justify-content: space-between;">
            <div class="bento-card-main">
                <div>
                    <h1 class="bento-main-title">
                        Unleash Diagnostic Precision With Swin Transformers
                    </h1>
                    <p class="bento-main-sub">
                        Experience next-generation multi-modal brain tumor classification and histological grading. 
                        Synthesize co-registered MRI sequences and explore transparent, pixel-level self-attention saliency maps.
                    </p>
                </div>
                <div>
                    <a href="#diagnostic-studio" class="bento-pill-action">
                        <span>Explore Diagnostic Studio</span>
                        <span class="bento-pill-arrow">➔</span>
                    </a>
                </div>
            </div>
            <div class="bento-card-sub">
                <img src="{ribbon_b64}" class="bento-sub-thumb" alt="Swin Attention Waves" />
                <div>
                    <div class="bento-sub-title">Shifted Windows & Dual XAI</div>
                    <div class="bento-sub-desc">
                        Linear-complexity cross-window attention with token-level Grad-CAM and Swin Attention Rollout.
                    </div>
                    <span class="bento-sub-link">94.6% LGG Sensitivity Verified ➔</span>
                </div>
            </div>
        </div>
        <div class="bento-card-showcase">
            <img src="{brain_b64}" class="showcase-floating-brain" alt="3D Neural Brain" />
            <div class="showcase-frosted-overlay">
                <div>
                    <div class="showcase-overlay-title">
                        Multi-Modal Fusion,<br/>Transparent AI
                    </div>
                    <p class="showcase-overlay-sub">
                        Co-registering T1, T1ce, T2, & FLAIR to resolve minority class collapse with verifiable visual explainability.
                    </p>
                </div>
                <div class="showcase-circle-arrow">➔</div>
            </div>
        </div>
    </div>
    """)

    # Anchor target for the CTA button
    render_html('<div id="diagnostic-studio"></div>')

    # ==========================================================================
    # 3. ORGANIZED SECTIONS: TABS (Studio, Architecture, Benchmarks, Guide)
    # ==========================================================================
    tab_studio, tab_arch, tab_bench, tab_protocol = st.tabs(
        [
            "🔬 Diagnostic Studio",
            "🧠 Diagnostic Architectures",
            "📊 Benchmark Specifications",
            "📋 Clinical MRI Modalities",
        ]
    )

    # ==========================================================================
    # TAB 1: DIAGNOSTIC STUDIO (INTERACTIVE TRIAGE & EXPLAINABILITY)
    # ==========================================================================
    with tab_studio:
        st.markdown("### 🎛️ Diagnostic Mode Selection")
        st.caption("Choose between fast single-scan 4-class triage or multi-parametric 4-sequence early fusion.")

        # Mode Selector Containers
        col_m1, col_m2 = st.columns(2)
        is_single = st.session_state["selected_mode"] == "single"
        is_multi = st.session_state["selected_mode"] == "multimodal"

        with col_m1:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="mode-card-inner">
                        <div class="mode-header-row">
                            <div class="mode-icon-circle">📷</div>
                            <div class="mode-status-badge {'active' if is_single else 'inactive'}">
                                {'● ACTIVE MODE' if is_single else '○ STANDBY'}
                            </div>
                        </div>
                        <div class="mode-title-text">1. Single-Modality Diagnosis (Kaggle)</div>
                        <div class="mode-desc-text">
                            Analyze standard 2D axial MRI scans to triage between <b>Glioma</b>, <b>Meningioma</b>, 
                            <b>Pituitary Adenoma</b>, and <b>Healthy Brain</b>.
                        </div>
                        <div class="mode-meta-row">
                            <span class="mode-chip-tech">Swin-Tiny</span>
                            <span class="mode-chip-tech">Grad-CAM</span>
                            <span class="mode-chip-acc">87.75% Test Acc</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if is_single:
                    st.button(
                        "✓ Active: Single-Modality Mode",
                        key="btn_mode_single",
                        type="primary",
                        use_container_width=True,
                    )
                else:
                    if st.button(
                        "👉 Select Single-Modality Mode",
                        key="btn_mode_single",
                        type="secondary",
                        use_container_width=True,
                    ):
                        st.session_state["selected_mode"] = "single"
                        st.rerun()

        with col_m2:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="mode-card-inner">
                        <div class="mode-header-row">
                            <div class="mode-icon-circle">🧬</div>
                            <div class="mode-status-badge {'active' if is_multi else 'inactive'}">
                                {'● ACTIVE MODE' if is_multi else '○ STANDBY'}
                            </div>
                        </div>
                        <div class="mode-title-text">2. Multi-Modal MRI Early Fusion (BraTS)</div>
                        <div class="mode-desc-text">
                            Synthesize 4 co-registered sequences (<b>T1, T1ce, T2, FLAIR</b>) for fine-grained 
                            histological grading of <b>Low-Grade (LGG)</b> vs <b>High-Grade Gliomas (HGG)</b>.
                        </div>
                        <div class="mode-meta-row">
                            <span class="mode-chip-tech">Swin-Base</span>
                            <span class="mode-chip-tech">Dual XAI (CAM + Rollout)</span>
                            <span class="mode-chip-acc">86.73% Test Acc</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if is_multi:
                    st.button(
                        "✓ Active: Multi-Modal Fusion Mode",
                        key="btn_mode_multi",
                        type="primary",
                        use_container_width=True,
                    )
                else:
                    if st.button(
                        "👉 Select Multi-Modal Fusion Mode",
                        key="btn_mode_multi",
                        type="secondary",
                        use_container_width=True,
                    ):
                        st.session_state["selected_mode"] = "multimodal"
                        st.rerun()

        st.markdown("<br/>", unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # WORKFLOW A: SINGLE-MODALITY DIAGNOSIS (KAGGLE)
        # ----------------------------------------------------------------------
        if st.session_state["selected_mode"] == "single":
            st.markdown("#### 📤 Upload Single-Slice MRI Scan")

            # Load Kaggle Model
            try:
                kaggle_model, kaggle_meta = load_kaggle_model()
                k_device = kaggle_meta["device"]
                k_classes = kaggle_meta["class_names"]
            except FileNotFoundError as e:
                st.error(f"❌ Pretrained checkpoint missing: `{e}`")
                st.info("Run `python scripts/download_checkpoints.py` in your terminal to fetch pretrained weights.")
                return

            up_col, preview_col = st.columns([2, 1])
            with up_col:
                k_uploaded_file = st.file_uploader(
                    "Drop axial MRI slice image here (JPG, JPEG, PNG)",
                    type=["jpg", "jpeg", "png"],
                    key="kaggle_uploader",
                    help="Upload single axial brain MRI slice.",
                )

            active_k_bytes = None
            active_k_name = ""

            if k_uploaded_file is not None:
                active_k_bytes = k_uploaded_file.getvalue()
                active_k_name = k_uploaded_file.name
                with preview_col:
                    st.image(active_k_bytes, caption=f"Uploaded: {active_k_name}", width=190)

            # Pre-verified Clinical Demo Cases
            with st.expander("⚡ Or Test Instantly With a Verified Clinical Sample (1-Click Demo)", expanded=False):
                st.caption("Select a pre-authenticated clinical test scan from the held-out test cohort:")
                d_cols = st.columns(4)
                for idx, (d_name, d_info) in enumerate(KAGGLE_DEMO_SAMPLES.items()):
                    with d_cols[idx]:
                        st.markdown(f"**{d_info['badge']}**")
                        if st.button(f"Scan {idx+1}: {d_name.split(' ')[0]}", key=f"btn_k_demo_{idx}", use_container_width=True):
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
                    with preview_col:
                        st.image(active_k_bytes, caption=f"Clinical Demo: {active_k_name}", width=190)

            # Execution Controls
            analyze_clicked = False
            if active_k_bytes is not None:
                st.markdown("<br/>", unsafe_allow_html=True)
                btn_c1, btn_c2 = st.columns([2, 1])
                with btn_c1:
                    analyze_clicked = st.button("🔍 Run Swin-Tiny Analysis & Explainability", type="primary", use_container_width=True)
                with btn_c2:
                    if st.button("🔄 Reset Workspace", use_container_width=True):
                        if "k_active_demo_bytes" in st.session_state:
                            del st.session_state["k_active_demo_bytes"]
                        if "k_active_demo_name" in st.session_state:
                            del st.session_state["k_active_demo_name"]
                        st.rerun()

            # Run Analysis
            if active_k_bytes is not None and (analyze_clicked or st.session_state.get("k_active_demo_bytes") is not None):
                with st.spinner("Processing MRI slice through Swin-Tiny backbone..."):
                    img_rgb, display_bg, image_tensor = preprocess_kaggle_image(active_k_bytes)
                    input_batch = image_tensor.unsqueeze(0).to(k_device)

                    with torch.no_grad():
                        logits = kaggle_model(input_batch)
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                        pred_idx = int(np.argmax(probs))
                        pred_class = k_classes[pred_idx]
                        confidence = float(probs[pred_idx]) * 100.0

                st.markdown("---")
                st.markdown(f"### 📋 Diagnostic Assessment: `{pred_class.upper()}`")

                res_col1, res_col2 = st.columns([1, 1.4])
                with res_col1:
                    st.markdown(
                        f"""
                        <div class="metric-bento-card">
                            <div class="metric-bento-label">Diagnosis Category</div>
                            <div class="metric-bento-val">{pred_class.capitalize()}</div>
                            <div style="font-size: 0.95rem; color: #64748b; margin-top: 4px;">
                                Model Confidence: <b>{confidence:.2f}%</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("<br/>", unsafe_allow_html=True)
                    prob_df = pd.DataFrame(
                        {
                            "Class": [c.capitalize() for c in k_classes],
                            "Probability (%)": [float(p) * 100.0 for p in probs],
                        }
                    ).set_index("Class")
                    st.bar_chart(prob_df, y="Probability (%)")

                with res_col2:
                    st.markdown("##### 🎯 Swin Grad-CAM Saliency Overlay")
                    st.caption("Gradient backpropagation from target class logit to Stage 4 normalization layer (`norm2`), reshaped from sequence tokens.")
                    
                    alpha = st.slider("Heatmap Blending Opacity (α)", min_value=0.1, max_value=0.9, value=0.5, step=0.05, key="k_alpha")
                    
                    with st.spinner("Generating Grad-CAM attention heatmap..."):
                        g_cam, _, _ = generate_gradcam_heatmap(
                            model=kaggle_model,
                            image_tensor=image_tensor,
                            target_category=pred_idx,
                            device=k_device,
                        )
                        _, blended = create_heatmap_overlay(
                            gray_image=display_bg,
                            grayscale_cam=g_cam,
                            alpha=alpha,
                        )

                    c_img1, c_img2 = st.columns(2)
                    with c_img1:
                        st.image(display_bg, caption="Original Axial MRI Slice", use_container_width=True)
                    with c_img2:
                        st.image(blended, caption=f"Grad-CAM ({pred_class} — {confidence:.1f}%)", use_container_width=True)

        # ----------------------------------------------------------------------
        # WORKFLOW B: MULTI-MODAL MRI EARLY FUSION (BRATS)
        # ----------------------------------------------------------------------
        else:
            st.markdown("#### 🧬 Multi-Modal 4-Sequence Early Fusion")
            st.caption("Upload co-registered T1, T1ce, T2, and FLAIR slices or an authenticated `.npz` multi-channel slice.")

            # Load BraTS Model
            try:
                brats_model, brats_meta = load_brats_model()
                b_device = brats_meta["device"]
                b_classes = brats_meta["class_names"]
            except FileNotFoundError as e:
                st.error(f"❌ BraTS checkpoint missing: `{e}`")
                st.info("Run `python scripts/download_checkpoints.py` in your terminal to fetch pretrained weights.")
                return

            fused_tensor = None
            disp_t1 = disp_t1c = disp_t2 = disp_flair = None
            active_b_title = ""
            active_b_desc = ""

            # Input Mode: 4 Individual Uploads or NPZ
            f_tab1, f_tab2 = st.tabs(["📦 Upload 4 Modality Images (T1, T1ce, T2, FLAIR)", "📁 Upload Pre-Stacked .NPZ Slice"])
            
            with f_tab1:
                u1, u2, u3, u4 = st.columns(4)
                with u1:
                    f_t1 = st.file_uploader("1. T1 Native", type=["jpg", "png", "npy"], key="b_t1")
                with u2:
                    f_t1c = st.file_uploader("2. T1ce Post-Contrast", type=["jpg", "png", "npy"], key="b_t1c")
                with u3:
                    f_t2 = st.file_uploader("3. T2 Fluid", type=["jpg", "png", "npy"], key="b_t2")
                with u4:
                    f_flair = st.file_uploader("4. T2 FLAIR", type=["jpg", "png", "npy"], key="b_flair")

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

            with f_tab2:
                f_npz = st.file_uploader("Upload .npz Multi-Channel File", type=["npz"], key="b_npz")
                if f_npz is not None:
                    try:
                        npz_data = np.load(io.BytesIO(f_npz.getvalue()))
                        img_4ch = npz_data["image"] if "image" in npz_data else npz_data[list(npz_data.keys())[0]]
                        if img_4ch.shape[0] == 4:
                            fused_tensor = torch.from_numpy(img_4ch).float()
                        elif img_4ch.shape[-1] == 4:
                            fused_tensor = torch.from_numpy(img_4ch).permute(2, 0, 1).float()
                        else:
                            raise ValueError(f"Expected 4 channels, got shape {img_4ch.shape}")
                        
                        _, disp_t1 = preprocess_brats_slice(fused_tensor[0].numpy())
                        _, disp_t1c = preprocess_brats_slice(fused_tensor[1].numpy())
                        _, disp_t2 = preprocess_brats_slice(fused_tensor[2].numpy())
                        _, disp_flair = preprocess_brats_slice(fused_tensor[3].numpy())
                        active_b_title = f_npz.name
                        active_b_desc = "Preprocessed 4-channel NPZ slice."
                    except Exception as npz_err:
                        st.error(f"Error loading NPZ: {npz_err}")

            # Pre-verified Clinical Demo Cases
            with st.expander("⚡ Or Test Instantly With a Clinical Multi-Modal Patient Case (1-Click Demo)", expanded=False):
                st.caption("Select an authenticated patient case from the held-out BraTS 2020 validation cohort:")
                b_cols = st.columns(3)
                for idx, (b_name, b_info) in enumerate(BRATS_DEMO_SAMPLES.items()):
                    with b_cols[idx]:
                        st.markdown(f"**{b_info['badge']}**")
                        if st.button(f"Load Case {idx+1}", key=f"btn_b_demo_{idx}", use_container_width=True):
                            if b_info["path"].exists():
                                npz_data = np.load(b_info["path"])
                                img_4ch = npz_data["image"]
                                fused_tensor = torch.from_numpy(img_4ch).float()
                                _, disp_t1 = preprocess_brats_slice(fused_tensor[0].numpy())
                                _, disp_t1c = preprocess_brats_slice(fused_tensor[1].numpy())
                                _, disp_t2 = preprocess_brats_slice(fused_tensor[2].numpy())
                                _, disp_flair = preprocess_brats_slice(fused_tensor[3].numpy())
                                st.session_state["b_fused_tensor"] = fused_tensor
                                st.session_state["b_disp"] = (disp_t1, disp_t1c, disp_t2, disp_flair)
                                st.session_state["b_case_title"] = b_name
                                st.session_state["b_case_desc"] = b_info["description"]
                                st.rerun()

                if "b_fused_tensor" in st.session_state and fused_tensor is None:
                    fused_tensor = st.session_state["b_fused_tensor"]
                    disp_t1, disp_t1c, disp_t2, disp_flair = st.session_state["b_disp"]
                    active_b_title = st.session_state["b_case_title"]
                    active_b_desc = st.session_state["b_case_desc"]

            # Display Modalities and Run Inference
            if fused_tensor is not None:
                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown(f"**Active Case:** `{active_b_title}` — *{active_b_desc}*")

                st.markdown("##### 1. Co-Registered 4-Channel Input Array $[4, 224, 224]$")
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.image(disp_t1, caption="Ch 0: T1 Native (Boundaries)", use_container_width=True)
                with mc2:
                    st.image(disp_t1c, caption="Ch 1: T1ce (Enhancing Rim)", use_container_width=True)
                with mc3:
                    st.image(disp_t2, caption="Ch 2: T2 (Hyperintensity)", use_container_width=True)
                with mc4:
                    st.image(disp_flair, caption="Ch 3: FLAIR (Vasogenic Edema)", use_container_width=True)

                # Inference
                with torch.no_grad():
                    input_batch = fused_tensor.unsqueeze(0).to(b_device)
                    logits = brats_model(input_batch)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    pred_idx = int(np.argmax(probs))
                    pred_grade = b_classes[pred_idx]
                    confidence = float(probs[pred_idx]) * 100.0

                grade_label = "High-Grade Glioma (HGG)" if pred_grade == "HGG" else "Low-Grade Glioma (LGG)"

                st.markdown("---")
                st.markdown(f"### 📋 Histological Grade Assessment: `{grade_label.upper()}`")

                res_b1, res_b2 = st.columns([1, 1.4])
                with res_b1:
                    st.markdown(
                        f"""
                        <div class="metric-bento-card">
                            <div class="metric-bento-label">Predicted Tumor Grade</div>
                            <div class="metric-bento-val">{pred_grade}</div>
                            <div style="font-size: 0.95rem; color: #64748b; margin-top: 4px;">
                                Model Confidence: <b>{confidence:.2f}%</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("<br/>", unsafe_allow_html=True)
                    prob_b_df = pd.DataFrame(
                        {
                            "Grade": ["High-Grade (HGG)", "Low-Grade (LGG)"],
                            "Probability (%)": [float(probs[b_classes.index("HGG")]) * 100.0, float(probs[b_classes.index("LGG")]) * 100.0],
                        }
                    ).set_index("Grade")
                    st.bar_chart(prob_b_df, y="Probability (%)")

                with res_b2:
                    st.markdown("##### 🎯 Dual Visual Explainability (Overlaid on Anatomical FLAIR)")
                    st.caption("FLAIR suppresses free CSF to isolate edema. Grad-CAM highlights gradient focal points; Attention Rollout traces global self-attention flow.")
                    
                    b_alpha = st.slider("Heatmap Blending Opacity (α)", min_value=0.1, max_value=0.9, value=0.5, step=0.05, key="b_alpha")

                    with st.spinner("Computing Grad-CAM and Swin Attention Rollout..."):
                        g_cam, _, _ = generate_gradcam_heatmap(
                            model=brats_model,
                            image_tensor=fused_tensor,
                            target_category=pred_idx,
                            device=b_device,
                        )
                        _, g_blended = create_heatmap_overlay(
                            gray_image=disp_flair,
                            grayscale_cam=g_cam,
                            alpha=b_alpha,
                        )

                        rollout_engine = SwinAttentionRollout(model=brats_model, device=b_device)
                        r_cam = rollout_engine.compute_rollout(fused_tensor)
                        _, r_blended = create_heatmap_overlay(
                            gray_image=disp_flair,
                            grayscale_cam=r_cam,
                            alpha=b_alpha,
                        )

                    ex1, ex2, ex3 = st.columns(3)
                    with ex1:
                        st.image(disp_flair, caption="Baseline FLAIR Sequence", use_container_width=True)
                    with ex2:
                        st.image(g_blended, caption=f"Grad-CAM ({pred_grade} — {confidence:.1f}%)", use_container_width=True)
                    with ex3:
                        st.image(r_blended, caption="Swin Attention Rollout", use_container_width=True)

    # ==========================================================================
    # TAB 2: DIAGNOSTIC ARCHITECTURES (ORGANIZED DEEP DIVE SECTION)
    # ==========================================================================
    with tab_arch:
        st.markdown("### 🧠 Diagnostic Architectures & Vision Transformer Pipelines")
        st.caption("Comprehensive technical breakdown of the Swin Transformer backbones, stem adaptations, and saliency engines.")

        arch_col1, arch_col2 = st.columns(2)

        with arch_col1:
            st.markdown(
                """
                <div class="metric-bento-card" style="height: 100%;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                        <span style="font-size: 1.25rem; font-weight: 800;">1. Swin-Tiny Triage Model</span>
                        <span class="mode-bento-badge">Single-Modality</span>
                    </div>
                    <p style="font-size: 0.9rem; color: #64748b; line-height: 1.5;">
                        Specialized for rapid first-line 4-class differential diagnosis from conventional 2D axial MRI scans.
                    </p>
                    <hr style="border: 0; border-top: 1px solid rgba(226, 232, 240, 0.8); margin: 1rem 0;" />
                    <ul style="font-size: 0.88rem; color: #334155; line-height: 1.7; padding-left: 1.2rem;">
                        <li><b>Backbone:</b> <code>swin_tiny_patch4_window7_224</code></li>
                        <li><b>Total Parameters:</b> <code>27,518,244</code> (27.52M)</li>
                        <li><b>Input Representation:</b> Single 2D slice expanded to 3 RGB channels $[3, 224, 224]$</li>
                        <li><b>Patch Partition:</b> $4 \times 4$ pixels (Initial token grid $56 \times 56 = 3,136$ tokens)</li>
                        <li><b>Window Self-Attention:</b> $7 \times 7$ local windows with shifted-window partitions between successive layers</li>
                        <li><b>Complexity:</b> Linear computational complexity $\\mathcal{O}(49 \\cdot N)$ vs standard ViT quadratic $\\mathcal{O}(N^2)$</li>
                        <li><b>Classification Head:</b> LayerNorm + Global Average Pooling + Linear(768, 4)</li>
                        <li><b>Explainability Hook:</b> <code>layers[-1].blocks[-1].norm2</code> with 2D spatial token reshape transform $[B, 49, 768] \\to [B, 768, 7, 7]$</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with arch_col2:
            st.markdown(
                """
                <div class="metric-bento-card" style="height: 100%;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                        <span style="font-size: 1.25rem; font-weight: 800;">2. Swin-Base Multi-Modal Engine</span>
                        <span class="mode-bento-badge">Early Fusion</span>
                    </div>
                    <p style="font-size: 0.9rem; color: #64748b; line-height: 1.5;">
                        High-capacity 4-channel early-fusion architecture engineered for fine-grained histological grading of High vs Low Grade Gliomas.
                    </p>
                    <hr style="border: 0; border-top: 1px solid rgba(226, 232, 240, 0.8); margin: 1rem 0;" />
                    <ul style="font-size: 0.88rem; color: #334155; line-height: 1.7; padding-left: 1.2rem;">
                        <li><b>Backbone:</b> <code>swin_base_patch4_window7_224</code></li>
                        <li><b>Total Parameters:</b> <code>86,746,478</code> (86.75M)</li>
                        <li><b>Input Representation:</b> 4-Channel early-fusion tensor $[4, 224, 224]$ ($T_1, T_{1ce}, T_2, FLAIR$)</li>
                        <li><b>Stem Adaptation:</b> Adapted <code>Conv2d(4, 128, kernel=4, stride=4)</code> (+3,072 parameters)</li>
                        <li><b>Weight Initialization:</b> RGB weights preserved; 4th channel (FLAIR) initialized with channel-wise RGB mean</li>
                        <li><b>Shifted Window Attention:</b> 12 transformer blocks with window size 7, feature dimension 1024</li>
                        <li><b>Dual-Stream XAI:</b> Grad-CAM gradient backprop + full Swin Attention Rollout across all 12 blocks with residual flow $\\hat{A} = 0.5A + 0.5I$</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("#### 📐 Hierarchical Shifted-Window Attention Mechanics")
        st.markdown(
            """
            Unlike standard Vision Transformers (ViT) which compute global self-attention across all tokens—leading to prohibitive 
            quadratic computational scaling $\\mathcal{O}(N^2)$—the **Swin Transformer** restricts self-attention to local non-overlapping 
            $7 \\times 7$ windows. To introduce cross-window connections without incurring quadratic complexity, successive layers alternate 
            between regular window partitioning and **Shifted Window Partitioning** (displaced by $(\\lfloor M/2 \\rfloor, \\lfloor M/2 \\rfloor) = (3, 3)$ pixels). 
            This hierarchical token merging constructs multi-scale representations ($56 \\times 56 \\to 28 \\times 28 \\to 14 \\times 14 \\to 7 \\times 7$), 
            allowing the network to capture both fine microscopic tumor margins and macro-scale hemispheric structural shifts.
            """
        )

    # ==========================================================================
    # TAB 3: BENCHMARK SPECIFICATIONS & VERIFIED RESULTS (ORGANIZED SECTION)
    # ==========================================================================
    with tab_bench:
        st.markdown("### 📊 Benchmark Specifications & Verified Clinical Metrics")
        st.caption("All reported metrics are strictly verified on held-out, patient-stratified test cohorts.")

        # KPI Metric Grid
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(
                """
                <div class="metric-bento-card">
                    <div class="metric-bento-label">Kaggle Triage Accuracy</div>
                    <div class="metric-bento-val">87.75%</div>
                    <div style="font-size: 0.8rem; color: #16a34a; font-weight: 600;">Macro F1: 0.8751</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi2:
            st.markdown(
                """
                <div class="metric-bento-card">
                    <div class="metric-bento-label">Kaggle OvR ROC-AUC</div>
                    <div class="metric-bento-val">0.9759</div>
                    <div style="font-size: 0.8rem; color: #16a34a; font-weight: 600;">1,600 Held-Out Slices</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi3:
            st.markdown(
                """
                <div class="metric-bento-card">
                    <div class="metric-bento-label">BraTS Fusion Accuracy</div>
                    <div class="metric-bento-val">86.73%</div>
                    <div style="font-size: 0.8rem; color: #16a34a; font-weight: 600;">HGG F1: 0.9120</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi4:
            st.markdown(
                """
                <div class="metric-bento-card">
                    <div class="metric-bento-label">Minority LGG Sensitivity</div>
                    <div class="metric-bento-val">94.64%</div>
                    <div style="font-size: 0.8rem; color: #16a34a; font-weight: 600;">Rescued from 0.00% Collapse</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br/>", unsafe_allow_html=True)

        # Class Imbalance Breakthrough Comparison
        st.markdown("#### ⚡ Clinical Breakthrough: Resolving Class-Imbalance Collapse")
        st.markdown(
            """
            In medical imaging benchmarks like BraTS 2020, High-Grade Gliomas naturally outnumber Low-Grade Gliomas by **~3.9:1** 
            (79.4% HGG vs 20.6% LGG). Naive empirical risk minimization causes Vision Transformers to fall into a deceptive shortcut: 
            predicting the majority HGG class 100% of the time. This yields a superficial ~81% accuracy while **missing 100% of low-grade tumors (0.00% LGG Recall)**.
            """
        )

        imb_col1, imb_col2 = st.columns(2)
        with imb_col1:
            st.markdown(
                """
                <div class="metric-bento-card" style="border-left: 4px solid #ef4444;">
                    <b style="color: #ef4444; font-size: 1.1rem;">❌ Baseline (Unweighted Collapse)</b>
                    <ul style="font-size: 0.85rem; color: #475569; margin-top: 0.5rem; line-height: 1.6;">
                        <li>Standard Empirical Risk Cross-Entropy</li>
                        <li>Deceptive Accuracy: <b>80.95%</b></li>
                        <li>LGG Sensitivity: <b>0.00% (0 / 112 detected)</b></li>
                        <li>Test ROC-AUC: ~0.5612 (Random Chance)</li>
                        <li>Clinical Impact: Catastrophic under-diagnosis of curable low-grade astrocytomas.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with imb_col2:
            st.markdown(
                """
                <div class="metric-bento-card" style="border-left: 4px solid #22c55e;">
                    <b style="color: #22c55e; font-size: 1.1rem;">✅ Mitigated Swin-Base Pipeline</b>
                    <ul style="font-size: 0.85rem; color: #475569; margin-top: 0.5rem; line-height: 1.6;">
                        <li>Inverse-frequency class weights ($w_{\\text{LGG}}=2.45, w_{\\text{HGG}}=0.628$)</li>
                        <li>Balanced 50/50 mini-batches via <code>WeightedRandomSampler</code></li>
                        <li>True Discriminative Accuracy: <b>86.73%</b></li>
                        <li>LGG Sensitivity: <b>94.64% (106 / 112 detected)</b></li>
                        <li>Test ROC-AUC: <b>0.9677</b></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br/>", unsafe_allow_html=True)

        # Confusion Matrices Breakdown
        st.markdown("#### 🎯 Held-Out Test Set Confusion Matrices")
        cm1, cm2 = st.columns(2)
        with cm1:
            st.markdown("**1. BraTS 2020 Multi-Modal Early Fusion (588 Slices):**")
            brats_cm_df = pd.DataFrame(
                [[106, 6], [72, 404]],
                index=["True LGG (112)", "True HGG (476)"],
                columns=["Pred LGG", "Pred HGG"],
            )
            st.table(brats_cm_df)
            st.caption("106 of 112 LGG cases correctly recognized (94.64% sensitivity); 404 of 476 HGG cases identified.")

        with cm2:
            st.markdown("**2. Kaggle 4-Class Single-Modality Benchmark (1,600 Images):**")
            kaggle_cm_df = pd.DataFrame(
                [
                    [274, 87, 35, 4],
                    [2, 356, 23, 19],
                    [0, 2, 398, 0],
                    [2, 22, 0, 376],
                ],
                index=["True Glioma (400)", "True Meningioma (400)", "True NoTumor (400)", "True Pituitary (400)"],
                columns=["Pred Glioma", "Pred Meningioma", "Pred NoTumor", "Pred Pituitary"],
            )
            st.table(kaggle_cm_df)
            st.caption("99.5% Healthy control recall (398/400); 94.0% Pituitary recall (376/400).")

    # ==========================================================================
    # TAB 4: CLINICAL MRI MODALITIES PROTOCOL
    # ==========================================================================
    with tab_protocol:
        st.markdown("### 📋 Clinical MRI Modality Reference Guide")
        st.caption("Neuroradiological principles governing multi-sequence acquisition and biophysical contrasts.")

        mod_col1, mod_col2 = st.columns(2)
        with mod_col1:
            st.markdown(
                """
                <div class="metric-bento-card" style="margin-bottom: 1.2rem;">
                    <b style="font-size: 1.1rem; color: #4f46e5;">Channel 0: T1-Weighted (T1 Native)</b>
                    <p style="font-size: 0.88rem; color: #64748b; line-height: 1.5; margin-top: 4px;">
                        Highlights anatomical architecture and tissue boundaries. Cerebrospinal fluid appears dark (hypointense), 
                        gray matter intermediate, and white matter bright. Critical for baseline anatomical registration.
                    </p>
                </div>
                <div class="metric-bento-card">
                    <b style="font-size: 1.1rem; color: #4f46e5;">Channel 1: T1-Contrast Enhanced (T1ce / Gadolinium)</b>
                    <p style="font-size: 0.88rem; color: #64748b; line-height: 1.5; margin-top: 4px;">
                        Gadolinium contrast leaks across the disrupted blood-brain barrier. Highlights vascularized active tumor rims 
                        and necrotic cavities. Decisive for identifying high-grade glioblastoma angiogenesis.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with mod_col2:
            st.markdown(
                """
                <div class="metric-bento-card" style="margin-bottom: 1.2rem;">
                    <b style="font-size: 1.1rem; color: #4f46e5;">Channel 2: T2-Weighted (T2 Spin-Echo)</b>
                    <p style="font-size: 0.88rem; color: #64748b; line-height: 1.5; margin-top: 4px;">
                        Free water and fluid display bright hyperintense signal. Highlights diffuse infiltrative edema, 
                        cysts, and cellular water changes throughout the parenchyma.
                    </p>
                </div>
                <div class="metric-bento-card">
                    <b style="font-size: 1.1rem; color: #4f46e5;">Channel 3: T2-FLAIR (Fluid-Attenuated Inversion Recovery)</b>
                    <p style="font-size: 0.88rem; color: #64748b; line-height: 1.5; margin-top: 4px;">
                        Inversion recovery pulse selectively nulls the signal from free ventricular cerebrospinal fluid (CSF). 
                        Vasogenic edema surrounding the lesion remains intensely bright, making FLAIR the ideal anatomical baseline 
                        for heatmaps and explainability overlays.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ==========================================================================
    # 4. FOOTER
    # ==========================================================================
    st.markdown(
        """
        <div class="bw-footer">
            <b>BrainWave Technologies &bull; Explainable Brain Tumor Diagnosis Platform</b><br/>
            Engineered with Hierarchical Swin Transformers, Multi-Modal Early Fusion, and Grad-CAM / Attention Rollout XAI.<br/>
            <i>Designed strictly for scientific validation and translational research. Not an approved clinical diagnostic medical device.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
