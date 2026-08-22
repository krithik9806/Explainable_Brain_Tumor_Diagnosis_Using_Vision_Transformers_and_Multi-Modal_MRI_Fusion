"""
Full Training Loop for Vision Transformer & MRI Fusion Pipeline.

This script executes the full training run for single-modality brain tumor classification:
1. Loads configuration from YAML (configs/kaggle_config.yaml).
2. Builds full train and validation PyTorch DataLoaders (Kaggle dataset).
3. Instantiates SwinClassifier model (input_channels=3, num_classes=4).
4. Executes training loop with AdamW optimizer, Cosine Annealing LR scheduler, and Cross-Entropy loss.
5. Tracks metrics: train_loss, val_loss, val_accuracy per epoch.
6. Saves best model checkpoint as checkpoints/kaggle_best_model.pth (and inside checkpoints/kaggle/).
7. Generates and saves training loss curve plot to results/loss_curve.png.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import KaggleDataset
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
) -> Tuple[float, float]:
    """
    Evaluates the model on validation data with explicit eval mode and no_grad context.
    """
    model.eval()
    assert not model.training, "Model must be in evaluation mode (model.eval()) during validation"

    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            preds = torch.argmax(outputs, dim=1)
            correct_preds += torch.sum(preds == labels).item()
            total_samples += batch_size

    val_loss = running_loss / max(total_samples, 1)
    val_acc = correct_preds / max(total_samples, 1)
    return val_loss, val_acc


def plot_loss_curve(history: Dict[str, list], output_path: Path):
    """
    Plots training vs validation loss curve and saves plot image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="s", linewidth=2)
    plt.title("Swin Transformer Kaggle Classification - Loss Curve")
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
    max_samples: int = None,
    debug: bool = False,
):
    """
    Main training execution function.
    """
    start_time = time.time()

    # 1. Load configuration
    cfg = load_config(config_path)
    print(f"=== Loaded Configuration from {config_path} ===", flush=True)
    print(f"Experiment Name: {cfg.experiment_name}", flush=True)
    print(f"Backbone: {cfg.model.backbone}", flush=True)
    print(f"Num Classes: {cfg.dataset.num_classes}", flush=True)

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
    learning_rate = cfg.training.learning_rate
    weight_decay = cfg.training.weight_decay
    metric_to_monitor = getattr(cfg.checkpointing, "metric_to_monitor", "val_loss")

    if debug:
        print("[DEBUG MODE] Overriding run settings for fast debug validation:", flush=True)
        if epochs_override is None:
            epochs = 2
        if max_samples is None:
            max_samples = 500
        print(f" -> Epochs: {epochs}, Max Samples: {max_samples}, Batch Size: {batch_size}", flush=True)

    # 2. Setup DataLoaders
    print("=== Loading Data ===", flush=True)
    csv_path = PROJECT_ROOT / "data" / "processed" / "kaggle_splits.csv"
    train_dataset = KaggleDataset(csv_path=csv_path, split="train")
    val_dataset = KaggleDataset(csv_path=csv_path, split="val")

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

    # 3. Instantiate Model
    print("=== Instantiating SwinClassifier ===", flush=True)
    model = SwinClassifier(
        backbone_name=cfg.model.backbone,
        input_channels=cfg.dataset.input_channels,
        num_classes=cfg.dataset.num_classes,
        pretrained=cfg.model.pretrained,
    )
    model = model.to(device)
    print(f"Model parameters: {model.num_parameters:,}", flush=True)

    # 4. Optimizer, Scheduler, Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Setup W&B logging
    run_name = f"{cfg.experiment_name}_debug" if debug else cfg.experiment_name
    wandb_run = setup_wandb_logging(
        project_name=cfg.logging.wandb_project_name,
        config=cfg,
        run_name=run_name,
    )

    # Ensure Checkpoint Directory
    save_dir = Path(cfg.checkpointing.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    top_checkpoints_dir = PROJECT_ROOT / "checkpoints"
    top_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Track metrics history and best model
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}

    # 6. Training Loop
    print(f"\n=== Starting Training ({epochs} Epochs) ===", flush=True)
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch}/{epochs}] - Current LR: {current_lr:.6f}", flush=True)

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step LR scheduler
        scheduler.step()
        next_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start_time

        # Update History
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{epochs}] ({epoch_duration:.1f}s) -> "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
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
            # Save inside checkpoints/kaggle/
            best_model_pt = save_dir / "best_model.pt"
            best_model_pth = save_dir / "best_model.pth"
            torch.save(checkpoint_dict, best_model_pt)
            torch.save(checkpoint_dict, best_model_pth)

            # Save top-level checkpoints/kaggle_best_model.pth per project spec
            top_best_pth = top_checkpoints_dir / "kaggle_best_model.pth"
            torch.save(checkpoint_dict, top_best_pth)

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
    plot_loss_curve(history, results_dir / "loss_curve.png")

    finish_wandb_logging()
    print(f"\n=== Training Completed Successfully in {time_str} ===", flush=True)
    print(f"Final Metrics -> Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Final Val Acc: {history['val_acc'][-1]*100:.2f}%", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train Swin Transformer on Brain MRI Dataset")
    parser.add_argument("--config", type=str, default="configs/kaggle_config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Max dataset samples for debug/fast run")
    parser.add_argument("--debug", action="store_true", help="Run short debug mode (2 epochs, 500 samples)")

    args = parser.parse_args()
    run_training(
        config_path=args.config,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        max_samples=args.max_samples,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
