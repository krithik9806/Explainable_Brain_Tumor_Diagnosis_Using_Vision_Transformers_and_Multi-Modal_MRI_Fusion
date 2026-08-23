"""
Day 22 Hyperparameter Tuning Experiment Suite for BraTS Multi-Modal Fusion.

Runs short comparison passes across 5 hyperparameter variations on BraTS dataset:
- Variation 0: Baseline (LR=1e-4, BS=16, WD=0.01, Aug=Default)
- Variation 1: Lower LR (LR=3e-5, BS=16, WD=0.01, Aug=Default)
- Variation 2: Higher Weight Decay (LR=1e-4, BS=16, WD=0.05, Aug=Default)
- Variation 3: Smaller Batch Size & Moderate LR (LR=5e-5, BS=8, WD=0.02, Aug=Default)
- Variation 4: Tuned LR & Weaker Augmentation (LR=5e-5, BS=16, WD=0.01, Aug=Weaker)

Logs metrics to console/W&B and compiles a summary comparison table.
"""

import sys
from pathlib import Path
import albumentations as A
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import run_training

# Custom transform for weaker augmentation variation
weaker_aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.2),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=5, p=0.3, border_mode=cv2.BORDER_CONSTANT),
])

def custom_transform_callable(fused_tensor):
    # PyTorch [4, H, W] tensor -> numpy [H, W, 4] -> albumentations -> PyTorch tensor
    np_img = fused_tensor.numpy().transpose(1, 2, 0)
    aug = weaker_aug_pipeline(image=np_img)["image"]
    return torch.from_numpy(aug.transpose(2, 0, 1))

VARIATIONS = [
    {
        "name": "Variation 0 (Baseline)",
        "lr": 0.0001,
        "batch_size": 16,
        "weight_decay": 0.01,
        "aug_desc": "Default (p=0.5, rot=15)",
        "transform": None,
        "run_name": "brats_tuning_var0_baseline",
    },
    {
        "name": "Variation 1 (Lower LR - 3e-5)",
        "lr": 0.00003,
        "batch_size": 16,
        "weight_decay": 0.01,
        "aug_desc": "Default",
        "transform": None,
        "run_name": "brats_tuning_var1_lower_lr",
    },
    {
        "name": "Variation 2 (Higher Weight Decay - 0.05)",
        "lr": 0.0001,
        "batch_size": 16,
        "weight_decay": 0.05,
        "aug_desc": "Default",
        "transform": None,
        "run_name": "brats_tuning_var2_high_wd",
    },
    {
        "name": "Variation 3 (Batch Size 8, Moderate LR - 5e-5)",
        "lr": 0.00005,
        "batch_size": 8,
        "weight_decay": 0.02,
        "aug_desc": "Default",
        "transform": None,
        "run_name": "brats_tuning_var3_bs8_mod_lr",
    },
    {
        "name": "Variation 4 (Tuned LR 5e-5 + Weaker Aug)",
        "lr": 0.00005,
        "batch_size": 16,
        "weight_decay": 0.01,
        "aug_desc": "Weaker (p=0.2, rot=5)",
        "transform": None,
        "run_name": "brats_tuning_var4_mod_lr_weak_aug",
    },
]


def run_all_variations(epochs: int = 3, max_samples: int = 240):
    results = []

    print("=" * 85)
    print(f"=== DAY 22 HYPERPARAMETER TUNING: BraTS MULTI-MODAL FUSION ({epochs} EPOCHS EACH) ===")
    print("=" * 85)

    for idx, var in enumerate(VARIATIONS, start=1):
        print(f"\n>>> Running [{idx}/{len(VARIATIONS)}]: {var['name']} ...", flush=True)
        res = run_training(
            config_path="configs/brats_fusion_config.yaml",
            epochs_override=epochs,
            batch_size_override=var["batch_size"],
            learning_rate_override=var["lr"],
            weight_decay_override=var["weight_decay"],
            run_name_override=var["run_name"],
            transform_override=var["transform"],
            max_samples=max_samples,
            debug=False,
        )

        res_record = {
            "variation": var["name"],
            "lr": var["lr"],
            "batch_size": var["batch_size"],
            "weight_decay": var["weight_decay"],
            "aug_desc": var["aug_desc"],
            "best_epoch": res["best_epoch"],
            "best_val_loss": res["best_val_loss"],
            "best_val_acc": res["best_val_acc"],
            "best_val_auc": res["best_val_auc"],
        }
        results.append(res_record)

    print("\n" + "=" * 90)
    print("=== HYPERPARAMETER TUNING SUMMARY COMPARISON TABLE ===")
    print("=" * 90)
    header = f"{'Variation':<42} | {'LR':<9} | {'BS':<4} | {'WD':<6} | {'Val Acc (%)':<11} | {'Val Loss':<9} | {'Val AUC':<8}"
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['variation']:<42} | {r['lr']:<9.5f} | {r['batch_size']:<4} | {r['weight_decay']:<6.2f} | "
            f"{r['best_val_acc']*100:<11.2f} | {r['best_val_loss']:<9.4f} | {r['best_val_auc']:<8.4f}"
        )
    print("=" * 90)

    return results


if __name__ == "__main__":
    run_all_variations(epochs=3, max_samples=240)

