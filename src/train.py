"""
Baseline Training Loop for Vision Transformer & MRI Fusion Pipeline.

This script implements the baseline training loop for single-modality brain tumor classification:
1. Loads configuration from YAML (e.g. configs/kaggle_config.yaml).
2. Initializes train and validation PyTorch DataLoaders.
3. Instantiates SwinClassifier model (input_channels=3, num_classes=4).
4. Executes training loop with AdamW optimizer, Cosine Annealing LR scheduler, and Cross-Entropy loss.
5. Logs epoch metrics (train_loss, val_loss, val_accuracy) to console and Weights & Biases (wandb).
6. Saves model checkpoints after every epoch to the configured checkpoint directory.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import KaggleDataset, get_dataloaders_from_config
from src.models.swin_model import SwinClassifier
from src.utils.config_loader import load_config
from src.utils.logging_setup import log_metrics, setup_wandb_logging


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Executes one training epoch over the given dataloader.

    Args:
        model: SwinClassifier model instance.
        dataloader: Training DataLoader.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: AdamW optimizer.
        device: Torch compute device (CPU or CUDA).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
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
    Evaluates the model on validation data.

    Args:
        model: SwinClassifier model instance.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Torch compute device.

    Returns:
        Tuple[float, float]: (validation_loss, validation_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in enumerate(dataloader) if isinstance(dataloader, list) else dataloader:
            if isinstance(images, int):  # fallback check
                continue
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
        print("For initial debugging, using a small subset of samples and few epochs is highly recommended.\n", flush=True)

    # Hyperparameters
    epochs = epochs_override if epochs_override is not None else cfg.training.num_epochs
    batch_size = batch_size_override if batch_size_override is not None else cfg.training.batch_size
    learning_rate = cfg.training.learning_rate
    weight_decay = cfg.training.weight_decay

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
        print(f"Subsetting data for debug run: Train={len(train_dataset)}, Val={len(val_dataset)}", flush=True)
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
    print(f"Checkpoint directory ready: {save_dir.resolve()}", flush=True)

    # 6. Training Loop
    print(f"\n=== Starting Training ({epochs} Epochs) ===", flush=True)
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch}/{epochs}] - LR: {current_lr:.6f}", flush=True)

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step LR scheduler
        scheduler.step()

        # Log metrics to console
        print(
            f"Epoch [{epoch}/{epochs}] Summary -> "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%",
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

        # 7. Save Checkpoint
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
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
        torch.save(checkpoint_dict, checkpoint_path)
        print(f" -> Checkpoint saved to: {checkpoint_path.resolve()}", flush=True)

    print("\n=== Training Completed Successfully ===", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train Swin Transformer on Brain MRI Dataset")
    parser.add_argument("--config", type=str, default="configs/kaggle_config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Max dataset samples for debug run")
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
