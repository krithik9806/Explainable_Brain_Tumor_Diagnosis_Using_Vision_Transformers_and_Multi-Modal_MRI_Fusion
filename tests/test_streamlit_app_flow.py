"""
Verification script for Streamlit app logic: model loading, preprocessing, inference, Grad-CAM, and error handling.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app import load_kaggle_model, preprocess_uploaded_image
from src.explain import create_heatmap_overlay, generate_gradcam_heatmap


def test_app_pipeline():
    print("=" * 70, flush=True)
    print("=== TESTING STREAMLIT APP INFERENCE & GRAD-CAM PIPELINE ===", flush=True)
    print("=" * 70, flush=True)

    # 1. Test model loading with backbone auto-detection
    print("\n[Step 1] Loading model using app.load_kaggle_model()...", flush=True)
    model, meta = load_kaggle_model()
    device = meta["device"]
    class_names = meta["class_names"]

    print(f"  -> Model successfully loaded!", flush=True)
    print(f"  -> Backbone: {meta['backbone_name']}", flush=True)
    print(f"  -> Classes: {class_names}", flush=True)
    print(f"  -> Device: {device}", flush=True)

    assert meta["backbone_name"] == "swin_tiny_patch4_window7_224", f"Expected swin_tiny, got {meta['backbone_name']}"
    assert len(class_names) == 4, f"Expected 4 classes, got {len(class_names)}"

    # 2. Test samples from 2 different classes
    test_samples = [
        ("glioma", PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "glioma" / "Te-gl_1.jpg"),
        ("pituitary", PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "pituitary" / "Te-pi_1.jpg"),
    ]

    for true_label, img_path in test_samples:
        print(f"\n[Step 2] Testing sample: {img_path.name} (True class: {true_label})", flush=True)
        assert img_path.exists(), f"Image path does not exist: {img_path}"

        with open(img_path, "rb") as f:
            file_bytes = f.read()

        # Test preprocessing
        raw_rgb, display_bg, img_tensor = preprocess_uploaded_image(file_bytes)
        print(f"  -> Decoded raw RGB shape: {raw_rgb.shape}", flush=True)
        print(f"  -> Display BG shape: {display_bg.shape}, min={display_bg.min():.2f}, max={display_bg.max():.2f}", flush=True)
        print(f"  -> Preprocessed tensor shape: {img_tensor.shape}, dtype={img_tensor.dtype}", flush=True)

        assert img_tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {img_tensor.shape}"
        assert display_bg.shape == (224, 224, 3), f"Expected (224, 224, 3), got {display_bg.shape}"

        # Test inference
        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            conf = probs[pred_idx] * 100.0

        print(f"  -> Predicted class: {pred_class} (Confidence: {conf:.2f}%)", flush=True)
        print("  -> Probability breakdown:", flush=True)
        for cname, p in zip(class_names, probs):
            print(f"       * {cname:<12}: {p * 100:.2f}%", flush=True)

        # Test Grad-CAM
        print("  -> Computing Grad-CAM heatmap...", flush=True)
        grayscale_cam, cam_pred_idx, cam_conf = generate_gradcam_heatmap(
            model=model,
            image_tensor=img_tensor,
            target_category=pred_idx,
            device=device,
        )
        print(f"  -> Grad-CAM grayscale map shape: {grayscale_cam.shape}, min={grayscale_cam.min():.4f}, max={grayscale_cam.max():.4f}", flush=True)
        assert grayscale_cam.shape == (224, 224), f"Expected (224, 224), got {grayscale_cam.shape}"

        heatmap_rgb, blended_rgb = create_heatmap_overlay(display_bg, grayscale_cam, alpha=0.5)
        print(f"  -> Blended overlay shape: {blended_rgb.shape}, min={blended_rgb.min():.4f}, max={blended_rgb.max():.4f}", flush=True)
        assert blended_rgb.shape == (224, 224, 3), f"Expected (224, 224, 3), got {blended_rgb.shape}"

    # 3. Test Error Handling for invalid/corrupted upload
    print("\n[Step 3] Testing graceful error handling for invalid bytes...", flush=True)
    corrupted_bytes = b"NOT_A_VALID_IMAGE_FILE_JUST_RANDOM_TEXT_DATA"
    try:
        preprocess_uploaded_image(corrupted_bytes)
        raise AssertionError("Failed: preprocess_uploaded_image should have raised ValueError on corrupted bytes!")
    except ValueError as e:
        print(f"  -> Successfully caught expected error: {e}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("=== ALL STREAMLIT PIPELINE TESTS PASSED SUCCESSFULLY! ===", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    test_app_pipeline()
