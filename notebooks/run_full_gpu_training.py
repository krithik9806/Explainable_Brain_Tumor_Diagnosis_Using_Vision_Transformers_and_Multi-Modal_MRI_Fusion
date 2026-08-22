"""
Standalone GPU Execution Script for Full Kaggle 50-Epoch Swin Transformer Training.

Instructions for Google Colab / Kaggle Notebook / GPU Instance:
1. Clone repo or upload project files.
2. Install requirements: `pip install timm PyYAML pandas matplotlib wandb torch`
3. Execute script: `python notebooks/run_full_gpu_training.py`
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import run_training

if __name__ == "__main__":
    print("=== Launching Full 50-Epoch GPU Training Run ===")
    run_training(
        config_path="configs/kaggle_config.yaml",
        epochs_override=50,
        batch_size_override=32,
        max_samples=None,  # Full 4,480 train / 1,120 val dataset
        debug=False,
    )
