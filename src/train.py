"""
Hardened Baseline Training Loop for Vision Transformer & MRI Fusion Pipeline.

This script executes and validates the baseline training loop for single-modality brain tumor classification:
1. Loads configuration from YAML (e.g. configs/kaggle_config.yaml).
2. Builds train and validation PyTorch DataLoaders.
3. Instantiates SwinClassifier model (input_channels=3, num_classes=4).
4. Executes training loop with AdamW optimizer, Cosine Annealing LR scheduler, and Cross-Entropy loss.
5. Performs explicit correctness assertions:
   - Separate train vs val loss/accuracy calculation with sample-weighted averaging.
   - Strict model.train() during training and model.eval() with torch.no_grad() during validation.
   - Per-step optimizer.zero_grad() to prevent unintended gradient accumulation.
   - Per-epoch scheduler.step() for proper learning rate decay.
6. Logs metrics to console and Weights & Biases (wandb).
7. Saves per-epoch checkpoints (checkpoint_epoch_{N}.pt / .pth) AND tracks/saves the best model (best_model.pt / .pth).
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

    Args:
        model: SwinClassifier model instance.
        dataloader: Training DataLoader.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: AdamW optimizer.
        device: Torch compute device (CPU or CUDA).

    Returns:
        float: Sample-weighted average training loss for the epoch.
    """
    # CORRECTNESS CHECK 1: Ensure model is explicitly set to training mode
    model.train()
    assert model.training, "Model must be in training mode (model.train()) during train_one_epoch"

    running_loss = 0.0
    total_samples = 0

    for step, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # CORRECTNESS CHECK 2: Zero gradients BEFORE forward pass on every mini-batch
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Check for NaN/Inf in loss
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

    Args:
        model: SwinClassifier model instance.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Torch compute device.

    Returns:
        Tuple[float, float]: (validation_loss, validation_accuracy)
    """
    # CORRECTNESS CHECK 3: Ensure model is explicitly set to evaluation mode
    model.eval()
    assert not model.training, "Model must be in evaluation mode (model.eval()) during validation"

    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    # CORRECTNESS CHECK 4: Disable gradient computation during evaluation
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

    # Track best model checkpoint
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0

    # 6. Training Loop
    print(f"\n=== Starting Training ({epochs} Epochs) ===", flush=True)
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch}/{epochs}] - Current LR: {current_lr:.6f}", flush=True)

        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # CORRECTNESS CHECK 5: Step learning rate scheduler AFTER each epoch
        scheduler.step()
        next_lr = optimizer.param_groups[0]["lr"]

        # Log metrics to console
        print(
            f"Epoch [{epoch}/{epochs}] Summary -> "
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

        # 7. Save Per-Epoch Checkpoint (support both .pt and .pth filename formats)
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
        print(f" -> Epoch {epoch} checkpoint saved: {epoch_ckpt_pt.name} & {epoch_ckpt_pth.name}", flush=True)

        # 8. Best Model Checkpoint Tracking
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
            best_model_pt = save_dir / "best_model.pt"
            best_model_pth = save_dir / "best_model.pth"
            torch.save(checkpoint_dict, best_model_pt)
            torch.save(checkpoint_dict, best_model_pth)
            print(
                f" -> [BEST MODEL UPDATED] Epoch {epoch} reached new best {metric_to_monitor}: "
                f"{val_loss if metric_to_monitor == 'val_loss' else val_acc * 100:.2f}. "
                f"Saved to {best_model_pt.name} & {best_model_pth.name}",
                flush=True,
            )

    finish_wandb_logging()
    print(f"\n=== Training Completed Successfully ===", flush=True)
    print(f"Best model was saved from Epoch {best_epoch} monitoring '{metric_to_monitor}'.", flush=True)


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
