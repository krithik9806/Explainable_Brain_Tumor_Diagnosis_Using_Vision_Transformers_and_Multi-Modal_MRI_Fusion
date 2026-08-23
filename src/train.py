"""
Full Config-Driven Training Loop for Vision Transformer & MRI Fusion Pipeline.

This script executes training runs for both single-modality (Kaggle) and multi-modal fusion (BraTS) MRI classification:
1. Loads configuration dynamically from YAML (configs/kaggle_config.yaml or configs/brats_fusion_config.yaml).
2. Builds train and validation PyTorch DataLoaders (BraTSDataset or KaggleDataset).
3. Supports automatic class imbalance mitigation (Class-Weighted Cross-Entropy Loss).
4. Instantiates SwinClassifier model matching input_channels (3 for Kaggle, 4 for BraTS), num_classes, and backbone choice.
5. Evaluates Accuracy and ROC AUC metrics per epoch.
6. Saves best model checkpoint into experiment save_dir and top-level checkpoints/.
7. Generates and saves training loss curve plots to results/.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import roc_auc_score

from src.data.datasets import BraTSDataset, KaggleDataset
from src.models.swin_model import SwinClassifier
from src.utils.config_loader import load_config
from src.utils.logging_setup import finish_wandb_logging, log_metrics, setup_wandb_logging


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Executes one training epoch with explicit mode verification and per-step zero_grad.
    """
    model.train()
    assert model.training, "Model must be in training mode (model.train()) during train_one_epoch"

    running_loss = 0.0
    total_samples = 0

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"NaN or Inf loss encountered at step {step}: loss={loss.item()}")

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / max(total_samples, 1)
    return epoch_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 2,
) -> Tuple[float, float, float]:
    """
    Evaluates the model on validation data with explicit eval mode, computing Loss, Accuracy, and ROC AUC.
    """
    model.eval()
    assert not model.training, "Model must be in evaluation mode (model.eval()) during validation"

    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            correct_preds += torch.sum(preds == labels).item()
            total_samples += batch_size

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / max(total_samples, 1)
    val_acc = correct_preds / max(total_samples, 1)

    # Compute ROC AUC
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    try:
        if num_classes == 2:
            val_auc = float(roc_auc_score(all_labels, all_probs[:, 1]))
        else:
            val_auc = float(roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro"))
    except Exception:
        val_auc = 0.5  # Fallback if single class present in subset

    return val_loss, val_acc, val_auc


def plot_loss_curve(history: Dict[str, list], output_path: Path, title: str):
    """
    Plots training vs validation loss curve and saves plot image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="s", linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Loss curve plot saved to: {output_path.resolve()}", flush=True)


def run_training(
    config_path: str = "configs/kaggle_config.yaml",
    epochs_override: int = None,
    batch_size_override: int = None,
    learning_rate_override: float = None,
    weight_decay_override: float = None,
    backbone_override: str = None,
    run_name_override: str = None,
    transform_override = None,
    max_samples: int = None,
    use_class_weights: bool = True,
    debug: bool = False,
):
    """
    Main config-driven training execution function supporting Kaggle & BraTS experiments.
    """
    start_time = time.time()

    # 1. Load configuration
    cfg = load_config(config_path)
    backbone = backbone_override if backbone_override is not None else cfg.model.backbone

    print(f"=== Loaded Configuration from {config_path} ===", flush=True)
    print(f"Experiment Name: {cfg.experiment_name}", flush=True)
    print(f"Dataset Name: {cfg.dataset.name}", flush=True)
    print(f"Backbone Architecture: {backbone}", flush=True)
    print(f"Input Channels: {cfg.dataset.input_channels}", flush=True)
    print(f"Num Classes: {cfg.dataset.num_classes} ({getattr(cfg.dataset, 'class_names', [])})", flush=True)

    # Determine device & report compute
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Compute Device Status ===", flush=True)
    if torch.cuda.is_available():
        print(f"Device: GPU ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print("Device: CPU-only", flush=True)
        print("WARNING: CUDA is not available. Training on CPU will be significantly slower.", flush=True)

    # Hyperparameters
    epochs = epochs_override if epochs_override is not None else cfg.training.num_epochs
    batch_size = batch_size_override if batch_size_override is not None else cfg.training.batch_size
    learning_rate = learning_rate_override if learning_rate_override is not None else cfg.training.learning_rate
    weight_decay = weight_decay_override if weight_decay_override is not None else cfg.training.weight_decay
    metric_to_monitor = getattr(cfg.checkpointing, "metric_to_monitor", "val_loss")

    if debug:
        print("[DEBUG MODE] Overriding run settings for fast debug validation:", flush=True)
        if epochs_override is None:
            epochs = 3
        if max_samples is None:
            max_samples = 300
        print(f" -> Epochs: {epochs}, Max Samples: {max_samples}, Batch Size: {batch_size}", flush=True)

    # 2. Setup DataLoaders dynamically based on dataset name
    print("=== Loading Data ===", flush=True)
    ds_name = cfg.dataset.name.lower()
    if "brats" in ds_name:
        csv_path = PROJECT_ROOT / "data" / "processed" / "brats_splits.csv"
        train_dataset = BraTSDataset(csv_path=csv_path, split="train", class_names=cfg.dataset.class_names, transform=transform_override)
        val_dataset = BraTSDataset(csv_path=csv_path, split="val", class_names=cfg.dataset.class_names)
        exp_prefix = f"brats_{'base' if 'base' in backbone else 'tiny'}"
    elif "kaggle" in ds_name:
        csv_path = PROJECT_ROOT / "data" / "processed" / "kaggle_splits.csv"
        train_dataset = KaggleDataset(csv_path=csv_path, split="train", class_names=cfg.dataset.class_names, transform=transform_override)
        val_dataset = KaggleDataset(csv_path=csv_path, split="val", class_names=cfg.dataset.class_names)
        exp_prefix = f"kaggle_{'base' if 'base' in backbone else 'tiny'}"
    else:
        raise ValueError(f"Unrecognized dataset name '{cfg.dataset.name}' in config {config_path}")

    # Class balance audit & dynamic class weighting setup
    class_weights_tensor = None
    if hasattr(train_dataset, "df"):
        label_col = "grade" if "grade" in train_dataset.df.columns else "class_name"
        counts_dict = train_dataset.df[label_col].value_counts().to_dict()
        print(f"Train Class Distribution ({len(train_dataset)} total): {counts_dict}", flush=True)

        class_names = cfg.dataset.class_names
        counts_list = [counts_dict.get(c_name, 0) for c_name in class_names]
        total_count = sum(counts_list)
        num_cls = len(class_names)

        # Inverse frequency weighting
        weights_np = [total_count / (num_cls * max(cnt, 1)) for cnt in counts_list]
        weights_np = np.array(weights_np, dtype=np.float32)

        ratio = max(counts_list) / max(min(counts_list), 1)
        if ratio > 2.5 and use_class_weights:
            class_weights_tensor = torch.tensor(weights_np, dtype=torch.float32).to(device)
            print(
                f" -> [CLASS WEIGHTING ENABLED] Imbalance Ratio {ratio:.2f}:1 detected. "
                f"Class Weights applied to Loss: {dict(zip(class_names, [round(w, 3) for w in weights_np]))}",
                flush=True,
            )

    if max_samples is not None and max_samples > 0:
        train_subset_indices = list(range(min(len(train_dataset), max_samples)))
        val_subset_indices = list(range(min(len(val_dataset), max_samples)))
        train_dataset = Subset(train_dataset, train_subset_indices)
        val_dataset = Subset(val_dataset, val_subset_indices)
        print(f"Subsetting data for run: Train={len(train_dataset)}, Val={len(val_dataset)}", flush=True)
    else:
        print(f"Full Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Instantiate SwinClassifier with selected backbone, input_channels, num_classes
    print("=== Instantiating SwinClassifier ===", flush=True)
    model = SwinClassifier(
        backbone_name=backbone,
        input_channels=cfg.dataset.input_channels,
        num_classes=cfg.dataset.num_classes,
        pretrained=cfg.model.pretrained,
    )
    model = model.to(device)
    print(f"Model parameters ({backbone}): {model.num_parameters:,}", flush=True)

    # Input shape sanity check
    sample_images, sample_labels = next(iter(train_loader))
    print(f"Input batch shape check: Images={sample_images.shape}, Labels={sample_labels.shape}", flush=True)

    # 4. Optimizer, Scheduler, Loss (with Class Weighting if enabled)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Setup W&B logging
    run_name = run_name_override if run_name_override is not None else (f"{cfg.experiment_name}_{backbone}_debug" if debug else f"{cfg.experiment_name}_{backbone}")
    wandb_run = setup_wandb_logging(
        project_name=cfg.logging.wandb_project_name,
        config=cfg,
        run_name=run_name,
    )

    # Ensure Checkpoint Directory
    save_dir = Path(cfg.checkpointing.save_dir) / backbone
    save_dir.mkdir(parents=True, exist_ok=True)
    top_checkpoints_dir = PROJECT_ROOT / "checkpoints"
    top_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Track metrics history and best model
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_auc = 0.0
    best_epoch = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    # 6. Training Loop
    print(f"\n=== Starting Training ({epochs} Epochs) ===", flush=True)
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch}/{epochs}] - Current LR: {current_lr:.6f}", flush=True)

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, device, num_classes=cfg.dataset.num_classes
        )

        # Step LR scheduler
        scheduler.step()
        next_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start_time

        # Update History
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch [{epoch}/{epochs}] ({epoch_duration:.1f}s) -> "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val AUC: {val_auc:.4f} | "
            f"Next LR: {next_lr:.6f}",
            flush=True,
        )

        # Log metrics to W&B
        log_metrics(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_auc": val_auc,
                "learning_rate": current_lr,
            },
            step=epoch,
        )

        # Save Per-Epoch Checkpoint
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "backbone": backbone,
            "config": dict(cfg),
        }

        epoch_ckpt_pt = save_dir / f"checkpoint_epoch_{epoch}.pt"
        epoch_ckpt_pth = save_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint_dict, epoch_ckpt_pt)
        torch.save(checkpoint_dict, epoch_ckpt_pth)

        # Best Model Tracking
        is_best = False
        if metric_to_monitor == "val_loss":
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best = True
        else:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                is_best = True

        if is_best:
            best_epoch = epoch
            best_val_acc = val_acc
            best_val_auc = val_auc

            # Save inside experiment save_dir
            best_model_pt = save_dir / "best_model.pt"
            best_model_pth = save_dir / "best_model.pth"
            torch.save(checkpoint_dict, best_model_pt)
            torch.save(checkpoint_dict, best_model_pth)

            # Save top-level checkpoints/{exp_prefix}_best_model.pth per project spec
            top_best_pth = top_checkpoints_dir / f"{exp_prefix}_best_model.pth"
            torch.save(checkpoint_dict, top_best_pth)
            if exp_prefix.startswith("brats"):
                # Also save standard top-level checkpoints/brats_best_model.pth
                torch.save(checkpoint_dict, top_checkpoints_dir / "brats_best_model.pth")

            print(
                f" -> [BEST MODEL UPDATED] Epoch {epoch} reached best {metric_to_monitor}: "
                f"{val_loss if metric_to_monitor == 'val_loss' else val_acc * 100:.2f}. "
                f"Saved to {top_best_pth.resolve()}",
                flush=True,
            )

    # Total Training Time
    total_training_time = time.time() - start_time
    hours, rem = divmod(total_training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {seconds:.1f}s" if hours > 0 else f"{int(minutes)}m {seconds:.1f}s"

    # Plot and save loss curve
    results_dir = PROJECT_ROOT / "results"
    plot_loss_curve(
        history,
        results_dir / f"{exp_prefix}_loss_curve.png",
        f"Swin ({backbone}) {cfg.experiment_name} - Loss Curve",
    )

    finish_wandb_logging()
    print(f"\n=== Training Completed Successfully in {time_str} ===", flush=True)
    print(
        f"Final Metrics ({backbone}) -> Best Epoch: {best_epoch} | "
        f"Best Val Loss: {best_val_loss:.4f} | "
        f"Best Val Acc: {best_val_acc * 100:.2f}% | "
        f"Best Val AUC: {best_val_auc:.4f}",
        flush=True,
    )
    return {
        "backbone": backbone,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_auc": best_val_auc,
        "training_time": time_str,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Swin Transformer on Brain MRI Dataset")
    parser.add_argument("--config", type=str, default="configs/kaggle_config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--backbone", type=str, default=None, help="Override Swin backbone (e.g. swin_tiny_patch4_window7_224 or swin_base_patch4_window7_224)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max dataset samples for debug/fast run")
    parser.add_argument("--no_class_weights", action="store_true", help="Disable automatic class-weighted loss")
    parser.add_argument("--debug", action="store_true", help="Run short debug mode (3 epochs, 300 samples)")

    args = parser.parse_args()
    run_training(
        config_path=args.config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        backbone_override=args.backbone,
        max_samples=args.max_samples,
        use_class_weights=not args.no_class_weights,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
