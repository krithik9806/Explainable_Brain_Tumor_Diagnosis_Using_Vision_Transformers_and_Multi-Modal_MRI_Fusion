"""
Helper script to download pretrained model checkpoints from GitHub Releases.
Saves checkpoints to the `checkpoints/` directory.
"""

import os
import sys
import urllib.request
from pathlib import Path

REPO_URL = "https://github.com/krithik9806/Explainable_Brain_Tumor_Diagnosis_Using_Vision_Transformers_and_Multi-Modal_MRI_Fusion"
RELEASE_TAG = "v1.0-models"
BASE_DOWNLOAD_URL = f"{REPO_URL}/releases/download/{RELEASE_TAG}"

MODELS = {
    "kaggle_best_model.pth": {
        "url": f"{BASE_DOWNLOAD_URL}/kaggle_best_model.pth",
        "description": "Swin-Tiny checkpoint (Kaggle 4-class single-modality triage, ~315 MB)",
        "expected_size_mb": 315,
    },
    "brats_best_model.pth": {
        "url": f"{BASE_DOWNLOAD_URL}/brats_best_model.pth",
        "description": "Swin-Base checkpoint (BraTS 2020 4-channel multi-modal fusion, ~993 MB)",
        "expected_size_mb": 993,
    },
}


def download_with_progress(url: str, dest_path: Path):
    temp_path = dest_path.with_suffix(".tmp")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            bar_len = 30
            filled_len = int(bar_len * percent // 100)
            bar = "=" * filled_len + "-" * (bar_len - filled_len)
            sys.stdout.write(f"\r  [{bar}] {percent:5.1f}% ({mb_downloaded:6.1f} / {mb_total:6.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, temp_path, reporthook=report_progress)
        print()
        if temp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            return True
    except Exception as e:
        print(f"\n  [Error] Failed to download {url}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def main():
    root_dir = Path(__file__).resolve().parent.parent
    checkpoints_dir = root_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" NeuroVision: Pretrained Checkpoint Downloader")
    print("=" * 70)
    print(f"Target directory: {checkpoints_dir}\n")

    for filename, info in MODELS.items():
        dest = checkpoints_dir / filename
        if dest.exists() and dest.stat().st_size > 1024 * 1024:
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"[OK] {filename} already exists ({size_mb:.1f} MB). Skipping.")
            continue

        print(f"Downloading {filename}...")
        print(f"  Description: {info['description']}")
        print(f"  URL: {info['url']}")
        success = download_with_progress(info["url"], dest)
        if success:
            print(f"  Successfully downloaded to {dest.name}!\n")
        else:
            print(f"  [Failed] Could not download {filename}.\n")

    print("Done checking model checkpoints.")


if __name__ == "__main__":
    main()
