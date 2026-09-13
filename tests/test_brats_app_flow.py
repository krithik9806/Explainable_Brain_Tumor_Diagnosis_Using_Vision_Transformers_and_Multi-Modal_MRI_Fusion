"""
End-to-end verification script for BraTS multi-modal fusion mode and Kaggle single-modality mode.
Tests real LGG and HGG slices from held-out BraTS test split.
"""

import os
import sys
from pathlib import Path

# Force UTF-8 encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app import (
    load_brats_model,
    load_kaggle_model,
    preprocess_brats_slice,
)
from src.attention_rollout import SwinAttentionRollout
from src.explain import create_heatmap_overlay, generate_gradcam_heatmap
from src.fusion.fusion import fuse_brats_modalities


def test_dual_mode_pipeline():
    print("=" * 80, flush=True)
    print("=== TESTING DUAL-MODE STREAMLIT APP LOGIC (KAGGLE & BRATS) ===", flush=True)
    print("=" * 80, flush=True)

    # 1. Verify Model Loading & Backbone Auto-Detection for BOTH models
    print("\n[Step 1] Verifying Backbone Auto-Detection for Both Models...", flush=True)
    k_model, k_meta = load_kaggle_model()
    print(f"  -> Kaggle Model Loaded: {k_meta['backbone_name']} ({k_meta['input_channels']} channels, {k_meta['num_classes']} classes)", flush=True)
    assert k_meta["backbone_name"] == "swin_tiny_patch4_window7_224", f"Expected swin_tiny, got {k_meta['backbone_name']}"

    b_model, b_meta = load_brats_model()
    b_device = b_meta["device"]
    b_classes = b_meta["class_names"]
    print(f"  -> BraTS Model Loaded:  {b_meta['backbone_name']} ({b_meta['input_channels']} channels, {b_meta['num_classes']} classes)", flush=True)
    assert b_meta["backbone_name"] == "swin_base_patch4_window7_224", f"Expected swin_base, got {b_meta['backbone_name']}"

    # 2. Test BraTS Test Set Samples (LGG and HGG)
    test_cases = [
        {
            "true_grade": "LGG",
            "file": PROJECT_ROOT / "data" / "processed" / "brats_normalized" / "BraTS20_Training_310" / "slice_052.npz",
        },
        {
            "true_grade": "HGG",
            "file": PROJECT_ROOT / "data" / "processed" / "brats_normalized" / "BraTS20_Training_020" / "slice_040.npz",
        },
    ]

    for tc in test_cases:
        grade = tc["true_grade"]
        slice_path = tc["file"]
        print(f"\n[Step 2] Testing BraTS Sample: {slice_path.name} (True Grade: {grade})", flush=True)
        assert slice_path.exists(), f"Slice file not found: {slice_path}"

        data = np.load(slice_path)
        t1, t1ce, t2, flair = data["t1"], data["t1ce"], data["t2"], data["flair"]

        norm_t1, disp_t1 = preprocess_brats_slice(t1)
        norm_t1c, disp_t1c = preprocess_brats_slice(t1ce)
        norm_t2, disp_t2 = preprocess_brats_slice(t2)
        norm_flair, disp_flair = preprocess_brats_slice(flair)

        # Early fusion
        fused_tensor = fuse_brats_modalities(
            t1=norm_t1,
            t1ce=norm_t1c,
            t2=norm_t2,
            flair=norm_flair,
            target_size=(224, 224),
            return_tensor=True,
        )
        print(f"  -> Fused tensor shape: {fused_tensor.shape}, dtype={fused_tensor.dtype}", flush=True)
        assert fused_tensor.shape == (4, 224, 224), f"Expected (4, 224, 224), got {fused_tensor.shape}"

        # Inference
        with torch.no_grad():
            logits = b_model(fused_tensor.unsqueeze(0).to(b_device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_grade = b_classes[pred_idx]
            conf = probs[pred_idx] * 100.0

        print(f"  -> Predicted Tumor Grade: {pred_grade} (Confidence: {conf:.2f}%)", flush=True)
        for cname, p in zip(b_classes, probs):
            print(f"       * {cname}: {p * 100:.2f}%", flush=True)

        # Explainability: Grad-CAM
        print("  -> Computing Grad-CAM heatmap...", flush=True)
        g_cam, _, _ = generate_gradcam_heatmap(
            model=b_model,
            image_tensor=fused_tensor,
            target_category=pred_idx,
            device=b_device,
        )
        print(f"       Grad-CAM map range: [{g_cam.min():.4f}, {g_cam.max():.4f}], std={g_cam.std():.4f}", flush=True)
        assert g_cam.shape == (224, 224)
        assert g_cam.max() > 0.0, "Grad-CAM map is completely blank!"

        _, g_blended = create_heatmap_overlay(disp_flair, g_cam, alpha=0.5)
        assert g_blended.shape == (224, 224, 3)

        # Explainability: Swin Attention Rollout
        print("  -> Computing Swin Attention Rollout heatmap...", flush=True)
        rollout_engine = SwinAttentionRollout(model=b_model, device=b_device)
        r_cam = rollout_engine.compute_rollout(fused_tensor)
        print(f"       Attention Rollout map range: [{r_cam.min():.4f}, {r_cam.max():.4f}], std={r_cam.std():.4f}", flush=True)
        assert r_cam.shape == (224, 224)
        assert r_cam.max() > 0.0, "Attention Rollout map is completely blank!"

        _, r_blended = create_heatmap_overlay(disp_flair, r_cam, alpha=0.5)
        assert r_blended.shape == (224, 224, 3)

    print("\n" + "=" * 80, flush=True)
    print("=== ALL BRATS AND DUAL-MODE TESTS PASSED SUCCESSFULLY! ===", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    test_dual_mode_pipeline()
