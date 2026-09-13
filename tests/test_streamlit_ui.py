"""
Test Streamlit App using Streamlit's built-in AppTest framework.
Simulates uploading an image into st.file_uploader and verifies all UI components.
"""

import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_ui():
    print("=" * 60, flush=True)
    print("=== TESTING APP VIA STREAMLIT AppTest FRAMEWORK ===", flush=True)
    print("=" * 60, flush=True)

    app_path = PROJECT_ROOT / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60)

    # 1. Initial run without upload
    print("\n[1] Running app without upload...", flush=True)
    at.run()
    assert not at.exception, f"App threw an exception on load: {at.exception}"
    print("  -> App Title rendered:", repr(at.title[0].value), flush=True)
    assert "Explainable Brain Tumor Diagnosis System" in at.title[0].value
    assert len(at.file_uploader) == 1
    print("  -> File uploader label:", at.file_uploader[0].label, flush=True)
    print("  -> Info notification:", at.info[0].value, flush=True)

    # 2. Upload pituitary test image
    test_img = PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "pituitary" / "Te-pi_1.jpg"
    print(f"\n[2] Simulating file upload: {test_img.name}...", flush=True)
    with open(test_img, "rb") as f:
        img_bytes = f.read()

    # Upload using Streamlit testing API
    at.file_uploader[0].upload("Te-pi_1.jpg", img_bytes)
    at.run()

    assert not at.exception, f"App threw an exception after upload: {at.exception}"
    print("  -> Upload success text:", at.success[0].value, flush=True)
    print("  -> Metrics count:", len(at.metric), flush=True)
    for m in at.metric:
        print(f"       * {m.label}: {m.value}", flush=True)

    # Check images rendered (original + Grad-CAM)
    print("  -> Images rendered count:", len(at.image), flush=True)
    for i, img in enumerate(at.image):
        cap = img.proto.caption if hasattr(img.proto, "caption") else "rendered"
        print(f"       * Image {i+1} caption: {cap}", flush=True)

    assert len(at.metric) >= 2, f"Expected at least 2 metrics, got {len(at.metric)}"
    assert len(at.image) >= 2, f"Expected at least 2 images (original and heatmap), got {len(at.image)}"

    print("\n" + "=" * 60, flush=True)
    print("=== STREAMLIT AppTest PASSED 100% CLEANLY! ===", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    test_streamlit_ui()
