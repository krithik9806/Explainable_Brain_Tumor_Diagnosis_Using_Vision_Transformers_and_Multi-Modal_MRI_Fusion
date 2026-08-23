"""
Evaluation script for Explainable Brain Tumor Diagnosis models.

This module:
1. Loads test split DataLoader ('test' split from kaggle_splits.csv or brats_splits.csv).
2. Loads trained model weights from specified checkpoint and sets model to .eval() mode.
3. Computes evaluation metrics on held-out test set:
   - Accuracy
   - Precision (macro-averaged for multi-class Kaggle, binary for BraTS)
   - Recall (macro-averaged for multi-class Kaggle, binary for BraTS)
   - F1-Score (macro-averaged for multi-class Kaggle, binary for BraTS)
   - AUC (One-vs-Rest macro-averaged for Kaggle 4-class, standard binary AUC for BraTS)
   - Confusion Matrix
4. Saves confusion matrix heatmap and ROC curve plots to results/ directory labeled with class names.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data.datasets import BraTSDataset, KaggleDataset
from src.models.swin_model import SwinClassifier
from src.utils.config_loader import load_config


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs model inference over the dataloader without gradient computation.

    Returns:
        y_true (np.ndarray): True target labels.
        y_pred (np.ndarray): Predicted class indices.
        y_prob (np.ndarray): Predicted class probabilities (softmax).
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    """
    Computes accuracy, precision, recall, f1-score, and AUC metrics.
    """
    acc = accuracy_score(y_true, y_pred)

    if num_classes == 2:
        prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        auc_val = roc_auc_score(y_true, y_prob[:, 1])
        auc_strategy = "binary"
    else:
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        auc_val = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        auc_strategy = "one-vs-rest macro-averaged"

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc": float(auc_val),
        "auc_strategy": auc_strategy,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """
    Generates and saves a seaborn/matplotlib confusion matrix heatmap.
    """
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 12, "weight": "bold"},
    )
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class", fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, pad=15, weight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "ROC Curve",
) -> None:
    """
    Generates and saves ROC curve plot (binary or multi-class OvR).
    """
    num_classes = len(class_names)
    plt.figure(figsize=(8, 6))

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2.5,
            label=f"{class_names[1]} vs {class_names[0]} (AUC = {roc_auc:.4f})",
        )
    else:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for i in range(num_classes):
            binary_y = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_y, y_prob[:, i])
            class_auc = auc(fpr, tpr)
            color = colors[i % len(colors)]
            plt.plot(
                fpr,
                tpr,
                lw=2,
                color=color,
                label=f"Class '{class_names[i]}' (AUC = {class_auc:.4f})",
            )
        title += f"\n(Macro-Averaged AUC = {macro_auc:.4f})"

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(title, fontsize=14, pad=15, weight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_evaluation(
    checkpoint_path: Union[str, Path],
    config_path: Union[str, Path],
    output_dir: Union[str, Path] = "results",
    save_prefix: Optional[str] = None,
) -> Dict[str, float]:
    """
    Loads model checkpoint & config, runs evaluation on held-out test split,
    computes metrics, and saves confusion matrix and ROC curve plots.
    """
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path)
    output_dir = Path(output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 1. Load Config
    cfg = load_config(config_path)
    ds_name = cfg.dataset.name.lower()
    num_classes = cfg.dataset.num_classes
    class_names = cfg.dataset.class_names
    input_channels = cfg.dataset.input_channels

    print(f"\n==================================================", flush=True)
    print(f"Evaluating Model Checkpoint: {checkpoint_path}", flush=True)
    print(f"Config: {config_path}", flush=True)
    print(f"Dataset: {ds_name} ({num_classes} classes: {class_names})", flush=True)
    print(f"==================================================", flush=True)

    # 2. Determine Test DataLoader
    if "brats" in ds_name:
        csv_path = PROJECT_ROOT / "data" / "processed" / "brats_splits.csv"
        test_dataset = BraTSDataset(csv_path=csv_path, split="test", class_names=class_names)
        default_prefix = "brats"
    elif "kaggle" in ds_name:
        csv_path = PROJECT_ROOT / "data" / "processed" / "kaggle_splits.csv"
        test_dataset = KaggleDataset(csv_path=csv_path, split="test", class_names=class_names)
        default_prefix = "kaggle"
    else:
        raise ValueError(f"Unsupported dataset name '{ds_name}' in config {config_path}")

    batch_size = cfg.training.batch_size
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    print(f"Loaded held-out TEST split: {len(test_dataset)} samples ({len(test_loader)} batches)", flush=True)

    # 3. Load Checkpoint & Instantiate Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    backbone_name = (
        ckpt.get("backbone")
        if isinstance(ckpt, dict) and "backbone" in ckpt
        else cfg.model.backbone
    )

    model = SwinClassifier(
        backbone_name=backbone_name,
        input_channels=input_channels,
        num_classes=num_classes,
        pretrained=False,
    )

    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # 4. Inference
    print("Running inference over test set...", flush=True)
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)

    # 5. Compute Metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=num_classes)

    prefix = save_prefix if save_prefix else default_prefix
    exp_title = f"{prefix.upper()} ({backbone_name})"

    # Print Formatted Results
    print("\n---------------- EVALUATION RESULTS ----------------", flush=True)
    print(f"Model Architecture: {backbone_name}", flush=True)
    print(f"Test Accuracy:    {metrics['accuracy'] * 100:.2f}% ({metrics['accuracy']:.4f})", flush=True)
    print(f"Test Precision:   {metrics['precision']:.4f}", flush=True)
    print(f"Test Recall:      {metrics['recall']:.4f}", flush=True)
    print(f"Test F1-Score:    {metrics['f1_score']:.4f}", flush=True)
    print(f"Test AUC ({metrics['auc_strategy']}): {metrics['auc']:.4f}", flush=True)
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    print("---------------------------------------------------\n", flush=True)

    # 6. Save Plots
    cm_path = output_dir / f"{prefix}_confusion_matrix.png"
    roc_path = output_dir / f"{prefix}_roc_curve.png"

    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        class_names=class_names,
        save_path=cm_path,
        title=f"Confusion Matrix - {exp_title}",
    )
    print(f"Saved confusion matrix plot to: {cm_path.resolve()}", flush=True)

    plot_roc_curves(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        save_path=roc_path,
        title=f"ROC Curve - {exp_title}",
    )
    print(f"Saved ROC curve plot to: {roc_path.resolve()}", flush=True)

    metrics["cm_path"] = str(cm_path)
    metrics["roc_path"] = str(roc_path)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Swin Transformer Brain Tumor Classifier on Test Set.")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        required=True,
        help="Path to trained PyTorch model checkpoint (.pth or .pt).",
    )
    parser.add_argument(
        "--config",
        "-cfg",
        type=str,
        required=True,
        help="Path to experiment configuration YAML file.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="results",
        help="Directory to save evaluation plot outputs (default: results/).",
    )
    parser.add_argument(
        "--save_prefix",
        "-p",
        type=str,
        default=None,
        help="Prefix for saved plot filenames (e.g., 'kaggle' or 'brats').",
    )

    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        save_prefix=args.save_prefix,
    )


if __name__ == "__main__":
    main()
