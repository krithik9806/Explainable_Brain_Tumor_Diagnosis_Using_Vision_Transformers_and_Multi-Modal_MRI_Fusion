"""
GPU-Accelerated Full 50-Epoch BraTS Multi-Modal Fusion Training Script.

Designed for execution on Google Colab or Kaggle GPU (T4 / P100 / A100).
Executes full 50-epoch training run for 4-channel Swin Transformer (Swin-Tiny and Swin-Base),
with automatic class-weighted loss mitigation and ROC AUC tracking.

Usage:
    python notebooks/run_brats_gpu_training.py --backbone swin_base_patch4_window7_224 --epochs 50
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import roc_auc_score

from src.data.datasets import BraTSDataset
from src.models.swin_model import SwinClassifier
from src.utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Full GPU Training for BraTS Multi-Modal Fusion")
    parser.add_argument("--config", type=str, default="configs/brats_fusion_config.yaml")
    parser.add_argument("--backbone", type=str, default="swin_base_patch4_window7_224")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== GPU Training Initialized on {device} ===")

    cfg = load_config(args.config)
    csv_path = PROJECT_ROOT / "data" / "processed" / "brats_splits.csv"

    train_ds = BraTSDataset(csv_path=csv_path, split="train", class_names=cfg.dataset.class_names)
    val_ds = BraTSDataset(csv_path=csv_path, split="val", class_names=cfg.dataset.class_names)

    # Class balance & weighting
    counts_dict = train_ds.df["grade"].value_counts().to_dict()
    counts_list = [counts_dict.get(c, 0) for c in cfg.dataset.class_names]
    weights = [sum(counts_list) / (len(counts_list) * max(cnt, 1)) for cnt in counts_list]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Class Weights Applied (LGG vs HGG): {dict(zip(cfg.dataset.class_names, weights))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SwinClassifier(
        backbone_name=args.backbone,
        input_channels=4,
        num_classes=2,
        pretrained=True,
    ).to(device)

    print(f"Model parameters ({args.backbone}): {model.num_parameters:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_auc = 0.0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    print(f"\nStarting {args.epochs} Epochs Training...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss = 0.0
        t_total = 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

            t_loss += loss.item() * imgs.size(0)
            t_total += imgs.size(0)

        train_loss = t_loss / max(t_total, 1)

        # Validation
        model.eval()
        v_loss = 0.0
        correct = 0
        v_total = 0
        all_lbls, all_probs = [], []

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                v_loss += loss.item() * imgs.size(0)

                probs = torch.softmax(out, dim=1)
                preds = torch.argmax(out, dim=1)
                correct += (preds == lbls).sum().item()
                v_total += imgs.size(0)

                all_lbls.extend(lbls.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss = v_loss / max(v_total, 1)
        val_acc = correct / max(v_total, 1)
        val_auc = float(roc_auc_score(np.array(all_lbls), np.array(all_probs)[:, 1]))

        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] -> Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val AUC: {val_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_auc = val_auc
            ckpt_path = PROJECT_ROOT / "checkpoints" / "brats_best_model.pth"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_auc": val_auc,
                    "backbone": args.backbone,
                },
                ckpt_path,
            )
            print(f" -> Best Checkpoint Saved: val_loss={val_loss:.4f}, val_auc={val_auc:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time/60:.2f} mins.")
    print(f"Best Metrics -> Val Loss: {best_val_loss:.4f} | Val Acc: {best_val_acc*100:.2f}% | Val AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
