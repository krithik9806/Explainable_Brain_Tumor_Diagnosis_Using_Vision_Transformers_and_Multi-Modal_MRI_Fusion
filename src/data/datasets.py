"""
PyTorch Dataset and DataLoader Classes for BraTS and Kaggle Brain MRI Datasets.

This module provides:
1. BraTSDataset: Loads preprocessed 4-modality BraTS 2D slice .npz files, applies early fusion
   (fuse_brats_modalities) into a [4, 224, 224] float32 tensor, and returns (image, label).
2. KaggleDataset: Loads preprocessed 3-channel Kaggle image .npz files, applies pass-through
   (pass_through_kaggle) into a [3, 224, 224] float32 tensor, and returns (image, label).
3. DataLoader Factory Functions: Construct train, validation, and test PyTorch DataLoaders
   with configurable batch sizes and splitting matching project configs.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.fusion.fusion import fuse_brats_modalities, pass_through_kaggle
from src.utils.config_loader import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_file_path(file_path: Union[str, Path]) -> Path:
    """
    Robustly resolves a dataset file path relative to project root or data directory.
    """
    p = Path(file_path)
    if p.is_absolute() and p.exists():
        return p

    # Check relative to project root
    p1 = PROJECT_ROOT / p
    if p1.exists():
        return p1

    # Check relative to PROJECT_ROOT / data
    p2 = PROJECT_ROOT / "data" / p
    if p2.exists():
        return p2

    raise FileNotFoundError(f"Unable to locate file at: {file_path} (checked {p1} and {p2})")


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS Multi-Modal 4-Channel MRI Slices.

    Labels:
        0: LGG (Low-Grade Glioma)
        1: HGG (High-Grade Glioma)
    """

    DEFAULT_CLASS_NAMES = ["LGG", "HGG"]

    def __init__(
        self,
        csv_path: Union[str, Path] = "data/processed/brats_splits.csv",
        split: str = "train",
        class_names: Optional[List[str]] = None,
        transform=None,
    ):
        """
        Args:
            csv_path: Path to brats_splits.csv mapping file.
            split: Data split to filter ('train', 'val', 'test').
            class_names: List of target class labels (default: ["LGG", "HGG"]).
            transform: Optional data augmentation transform callable.
        """
        super().__init__()
        self.csv_path = resolve_file_path(csv_path)
        self.split = split.lower()
        self.transform = transform
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Load and filter splits CSV
        df = pd.read_csv(self.csv_path)
        if "split" not in df.columns:
            raise KeyError(f"Expected 'split' column in {self.csv_path}")

        self.df = df[df["split"].str.lower() == self.split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No records found for split '{split}' in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        file_path = resolve_file_path(row["file_path"])
        grade_str = str(row["grade"]).strip()

        if grade_str not in self.class_to_idx:
            raise KeyError(f"Grade '{grade_str}' not in configured class_to_idx mapping: {self.class_to_idx}")
        label_idx = self.class_to_idx[grade_str]

        # Load .npz file
        npz_data = np.load(file_path)

        # Apply Day 11 Early Fusion to produce [4, 224, 224] tensor
        fused_tensor = fuse_brats_modalities(
            t1=npz_data["t1"],
            t1ce=npz_data["t1ce"],
            t2=npz_data["t2"],
            flair=npz_data["flair"],
            return_tensor=True,
        )

        if self.transform is not None:
            fused_tensor = self.transform(fused_tensor)

        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return fused_tensor, label_tensor


class KaggleDataset(Dataset):
    """
    PyTorch Dataset for Kaggle Single-Modality 3-Channel MRI Images.

    Labels:
        0: glioma
        1: meningioma
        2: notumor
        3: pituitary
    """

    DEFAULT_CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

    def __init__(
        self,
        csv_path: Union[str, Path] = "data/processed/kaggle_splits.csv",
        split: str = "train",
        class_names: Optional[List[str]] = None,
        transform=None,
    ):
        """
        Args:
            csv_path: Path to kaggle_splits.csv mapping file.
            split: Data split to filter ('train', 'val', 'test').
            class_names: List of target class labels.
            transform: Optional data augmentation transform callable.
        """
        super().__init__()
        self.csv_path = resolve_file_path(csv_path)
        self.split = split.lower()
        self.transform = transform
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Load and filter splits CSV
        df = pd.read_csv(self.csv_path)
        if "split" not in df.columns:
            raise KeyError(f"Expected 'split' column in {self.csv_path}")

        self.df = df[df["split"].str.lower() == self.split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No records found for split '{split}' in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        file_path = resolve_file_path(row["file_path"])
        class_str = str(row["class_name"]).strip()

        if class_str not in self.class_to_idx:
            raise KeyError(f"Class '{class_str}' not in configured class_to_idx mapping: {self.class_to_idx}")
        label_idx = self.class_to_idx[class_str]

        # Load .npz file
        npz_data = np.load(file_path)
        img_key = "image" if "image" in npz_data else list(npz_data.keys())[0]
        raw_img = npz_data[img_key]

        # Apply Day 11 Pass-Through to produce [3, 224, 224] tensor
        image_tensor = pass_through_kaggle(raw_img, return_tensor=True)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return image_tensor, label_tensor


def get_brats_dataloaders(
    csv_path: Union[str, Path] = "data/processed/brats_splits.csv",
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """
    Constructs PyTorch DataLoaders for BraTS train, val, and test splits.
    """
    dataloaders = {}
    for split in ["train", "val", "test"]:
        dataset = BraTSDataset(csv_path=csv_path, split=split)
        shuffle = split == "train"
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return dataloaders


def get_kaggle_dataloaders(
    csv_path: Union[str, Path] = "data/processed/kaggle_splits.csv",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """
    Constructs PyTorch DataLoaders for Kaggle train, val, and test splits.
    """
    dataloaders = {}
    for split in ["train", "val", "test"]:
        dataset = KaggleDataset(csv_path=csv_path, split=split)
        shuffle = split == "train"
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return dataloaders


def get_dataloaders_from_config(
    config_path: Union[str, Path],
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Constructs DataLoaders automatically according to configuration YAML file.
    """
    cfg = load_config(config_path)
    ds_name = cfg.dataset.name.lower()
    batch_size = cfg.training.batch_size

    if "brats" in ds_name:
        csv_path = PROJECT_ROOT / "data" / "processed" / "brats_splits.csv"
        return get_brats_dataloaders(csv_path=csv_path, batch_size=batch_size, num_workers=num_workers)
    elif "kaggle" in ds_name:
        csv_path = PROJECT_ROOT / "data" / "processed" / "kaggle_splits.csv"
        return get_kaggle_dataloaders(csv_path=csv_path, batch_size=batch_size, num_workers=num_workers)
    else:
        raise ValueError(f"Unsupported dataset name '{ds_name}' in config {config_path}")
