# Configuration Schema & Experiment Setup Guide

This directory contains the modular YAML configuration files for training and evaluating vision transformer models across two experiment workflows in the project **"Explainable Brain Tumor Diagnosis Using Vision Transformers and Multi-Modal MRI Fusion"**.

---

## 1. Directory Overview

| Config File | Experiment Scope | Description |
|---|---|---|
| [`base_config.yaml`](file:///c:/PROJECTS/Explainable_Brain_Tumor_Diagnosis_Using_Vision_Transformers_and_Multi-Modal_MRI_Fusion/configs/base_config.yaml) | Shared Defaults | Contains common default settings (logging, optimizer, scheduler, checkpointing, random seeds, augmentation defaults) referenced by specific experiments. |
| [`kaggle_config.yaml`](file:///c:/PROJECTS/Explainable_Brain_Tumor_Diagnosis_Using_Vision_Transformers_and_Multi-Modal_MRI_Fusion/configs/kaggle_config.yaml) | Experiment 1 (Kaggle) | 4-class single-modality MRI classification (`glioma`, `meningioma`, `notumor`, `pituitary`) using 2D T1-weighted scans (3-channel RGB-loaded). |
| [`brats_fusion_config.yaml`](file:///c:/PROJECTS/Explainable_Brain_Tumor_Diagnosis_Using_Vision_Transformers_and_Multi-Modal_MRI_Fusion/configs/brats_fusion_config.yaml) | Experiment 2 (BraTS) | Binary classification (`LGG` vs. `HGG`) using 4-channel multi-modal early fusion (T1, T1ce, T2, FLAIR) with patient-level split separation. |

---

## 2. Configuration Inheritance Architecture

To avoid parameter duplication across multiple configuration files, this project uses **Python Deep-Merge Inheritance via `base_config` reference**.

- Each experiment config declares a top-level key referencing the base config:
  ```yaml
  base_config: "configs/base_config.yaml"
  ```
- The config loader utility [`src/utils/config_loader.py`](file:///c:/PROJECTS/Explainable_Brain_Tumor_Diagnosis_Using_Vision_Transformers_and_Multi-Modal_MRI_Fusion/src/utils/config_loader.py) reads `base_config.yaml` first and recursively overlays experiment-specific fields.
- **Why loader deep-merge over standard YAML anchors?** Pure YAML anchors (`<<: *default`) only function within a single file in standard PyYAML. Cross-file loader merging allows clean, standard YAML files without custom non-standard parser tags.

---

## 3. Comparison of Experiments at the Config Level

| Configuration Section | Field | Experiment 1: Kaggle Single-Modality | Experiment 2: BraTS Multi-Modal Fusion |
|---|---|---|---|
| Meta | `experiment_name` | `kaggle_single_modality_swin` | `brats_multimodal_fusion_swin` |
| Dataset | `raw_data_path` | `data/raw/kaggle` | `data/raw/brats` |
| Dataset | `num_classes` | `4` | `2` |
| Dataset | `class_names` | `["glioma", "meningioma", "notumor", "pituitary"]` | `["LGG", "HGG"]` |
| Dataset | `input_channels` | `3` (RGB-loaded grayscale T1) | `4` (T1 + T1ce + T2 + FLAIR stack) |
| Model | `input_channel_override` | `null` (standard 3-channel stem) | `4` (modifies patch embedding stem) |
| Training | `batch_size` | `32` | `16` (reduced for 4-channel memory footprint) |
| Splits | `patient_level` | `false` (slice-level images) | `true` (enforces patient volume separation) |
| Checkpointing | `save_dir` | `checkpoints/kaggle` | `checkpoints/brats_fusion` |

---

## 4. Usage in Python Code

To load and parse any configuration file:

```python
from src.utils.config_loader import load_config

# Load Kaggle configuration
config = load_config("configs/kaggle_config.yaml")

# Access attributes via dot notation or dictionary indexing
print(config.experiment_name)              # "kaggle_single_modality_swin"
print(config.dataset.num_classes)           # 4
print(config.model.backbone)                # "swin_tiny_patch4_window7_224"
print(config['training']['learning_rate'])  # 0.0001
```
