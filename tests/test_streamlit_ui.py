"""
Test Streamlit App using Streamlit's built-in AppTest framework.
Simulates uploading images into both Kaggle single-modality and BraTS multi-modal tabs.
"""

import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_ui():
    print("=" * 60, flush=True)
    print("=== TESTING DUAL-MODE APP VIA STREAMLIT AppTest ===", flush=True)
    print("=" * 60, flush=True)

    app_path = PROJECT_ROOT / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60)

    # 1. Initial run without upload
    print("\n[1] Running app without upload...", flush=True)
    at.run()
    assert not at.exception, f"App threw an exception on load: {at.exception}"
    print("  -> App Title rendered:", repr(at.title[0].value), flush=True)
    assert "Explainable Brain Tumor Diagnosis System" in at.title[0].value

    # Check that 5 file uploaders are present (1 Kaggle + 4 BraTS)
    print("  -> Total file uploaders found:", len(at.file_uploader), flush=True)
    assert len(at.file_uploader) == 5

    # 2. Test Kaggle Single-Modality Mode Upload
    test_img_k = PROJECT_ROOT / "data" / "raw" / "kaggle" / "Testing" / "pituitary" / "Te-pi_1.jpg"
    print(f"\n[2] Testing Kaggle tab with {test_img_k.name}...", flush=True)
    with open(test_img_k, "rb") as f:
        img_bytes_k = f.read()

    at.file_uploader(key="kaggle_uploader").upload("Te-pi_1.jpg", img_bytes_k)
    at.run()
    assert not at.exception, f"App threw an exception on Kaggle upload: {at.exception}"

    # Verify Kaggle metrics
    print("  -> Kaggle metrics rendered:", [f"{m.label}: {m.value}" for m in at.metric if "Diagnostic" in m.label or "Confidence" in m.label], flush=True)

    # 3. Test BraTS Multi-Modal Mode Incomplete Input Validation
    print("\n[3] Testing BraTS tab with partial upload (only T1)...", flush=True)
    sample_npz = PROJECT_ROOT / "data" / "processed" / "brats_normalized" / "BraTS20_Training_020" / "slice_040.npz"
    data = np.load(sample_npz)

    import io
    def arr_to_npz_bytes(arr, key):
        buf = io.BytesIO()
        np.savez(buf, **{key: arr})
        return buf.getvalue()

    t1_bytes = arr_to_npz_bytes(data["t1"], "t1")
    t1c_bytes = arr_to_npz_bytes(data["t1ce"], "t1ce")
    t2_bytes = arr_to_npz_bytes(data["t2"], "t2")
    flair_bytes = arr_to_npz_bytes(data["flair"], "flair")

    # Upload only T1
    at.file_uploader(key="brats_t1").upload("t1.npz", t1_bytes)
    at.run()
    assert not at.exception, f"App threw an exception on partial BraTS upload: {at.exception}"

    # Verify warning is shown for incomplete upload
    warnings = [w.value for w in at.warning]
    print("  -> Incomplete upload warning:", warnings[-1] if warnings else "None", flush=True)
    assert any("Incomplete Input" in w for w in warnings), "Expected Incomplete Input warning!"

    # 4. Upload remaining 3 modalities (T1c, T2, FLAIR)
    print("\n[4] Uploading remaining 3 modalities (T1c, T2, FLAIR)...", flush=True)
    at.file_uploader(key="brats_t1c").upload("t1c.npz", t1c_bytes)
    at.file_uploader(key="brats_t2").upload("t2.npz", t2_bytes)
    at.file_uploader(key="brats_flair").upload("flair.npz", flair_bytes)
    at.run()
    assert not at.exception, f"App threw an exception on complete BraTS upload: {at.exception}"

    # Verify BraTS prediction and explainability
    brats_metrics = [f"{m.label}: {m.value}" for m in at.metric if "Grade" in m.label or "Confidence" in m.label]
    print("  -> BraTS metrics rendered:", brats_metrics, flush=True)
    print("  -> Total images rendered in app:", len(at.image), flush=True)

    # In BraTS mode: 4 input modalities + 3 explainability panels = 7 images
    assert len(at.image) >= 7, f"Expected at least 7 images rendered, got {len(at.image)}"

    print("\n" + "=" * 60, flush=True)
    print("=== DUAL-MODE STREAMLIT AppTest PASSED 100% CLEANLY! ===", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    test_streamlit_ui()
