# Raw Data Structure & Dataset Documentation

This document describes the exact directory structure, data formats, modalities, and quality findings for the raw datasets located under `data/raw/`.

---

## 1. Kaggle Brain Tumor MRI Dataset

**Location**: `data/raw/kaggle/`  
**Task Type**: 4-Class Classification (`glioma`, `meningioma`, `notumor`, `pituitary`)  
**Total Images**: 7,200 (Training: 5,600 | Testing: 1,600)  
**File Formats**: `.jpg`, `.png`

### Folder Hierarchy

```
data/raw/kaggle/
├── Testing/
│   ├── glioma/          (400 images, e.g., Te-gl_1.jpg)
│   ├── meningioma/      (400 images, e.g., Te-aug-me_1.jpg)
│   ├── notumor/         (400 images, e.g., Te-no_1.jpg)
│   └── pituitary/       (400 images, e.g., Te-pi_1.jpg)
└── Training/
    ├── glioma/          (1,400 images, e.g., Tr-gl_1.jpg)
    ├── meningioma/      (1,400 images, e.g., Tr-aug-me_1.jpg)
    ├── notumor/         (1,400 images, e.g., Tr-no_1.jpg)
    └── pituitary/       (1,400 images, e.g., Tr-pi_1.jpg)
```

### Dataset Statistics & Properties
- **Class Balance**:
  - **Training Set**: 1,400 images per class (25.0% glioma, 25.0% meningioma, 25.0% notumor, 25.0% pituitary).
  - **Testing Set**: 400 images per class (25.0% glioma, 25.0% meningioma, 25.0% notumor, 25.0% pituitary).
- **Image Dimensions**: Variable sizes across scans (e.g., 512x512, 491x624, 206x244; 447 unique resolution pairs).
- **Data Quality**: 0 corrupted or unreadable images found across all 7,200 samples.

---

## 2. BraTS 2020 Dataset (Kaggle Mirror: awsaf49/brats20-dataset-training-validation)

**Location**: `data/raw/brats/`  
**Task Type**: Multi-Modal MRI Segmentation & Survival Prediction  
**Total Patients**: 494 (369 Training | 125 Validation)  
**File Formats**: NIfTI (`.nii`)

### Folder Hierarchy

```
data/raw/brats/
├── BraTS2020_TrainingData/
│   └── MICCAI_BraTS2020_TrainingData/
│       ├── name_mapping.csv                      (369 rows x 6 columns)
│       ├── survival_info.csv                     (236 rows x 4 columns)
│       ├── BraTS20_Training_001/
│       │   ├── BraTS20_Training_001_flair.nii
│       │   ├── BraTS20_Training_001_seg.nii
│       │   ├── BraTS20_Training_001_t1.nii
│       │   ├── BraTS20_Training_001_t1ce.nii
│       │   └── BraTS20_Training_001_t2.nii
│       └── ... (369 patient directories: BraTS20_Training_001 to BraTS20_Training_369)
└── BraTS2020_ValidationData/
    └── MICCAI_BraTS2020_ValidationData/
        ├── name_mapping_validation_data.csv       (125 rows x 4 columns)
        ├── survival_evaluation.csv                (125 rows x 3 columns)
        ├── BraTS20_Validation_001/
        │   ├── BraTS20_Validation_001_flair.nii
        │   ├── BraTS20_Validation_001_t1.nii
        │   ├── BraTS20_Validation_001_t1ce.nii
        │   └── BraTS20_Validation_001_t2.nii
        └── ... (125 patient directories: BraTS20_Validation_001 to BraTS20_Validation_125)
```

### Modality Descriptions
- `*_t1.nii`: T1-weighted native MRI scan.
- `*_t1ce.nii`: T1-weighted post-contrast-enhanced MRI scan.
- `*_t2.nii`: T2-weighted native MRI scan.
- `*_flair.nii`: T2-Weighted Fluid-Attenuated Inversion Recovery (FLAIR) scan.
- `*_seg.nii`: Expert-annotated tumor sub-region segmentation mask (0 = Background, 1 = Necrotic and Non-Enhancing Tumor, 2 = Peritumoral Edema, 4 = GD-Enhancing Tumor).

### Dataset Statistics & Properties
- **Volume Shapes**: Consistent 3D voxel grid of `(240, 240, 155)` across all modalities.
- **Co-registration**: All 4 modalities for each patient are co-registered to the same anatomical space.
- **Class Distribution & Imbalance**:
  - Out of 369 training subjects: **293 HGG** (~79.4%) and **76 LGG** (~20.6%), presenting a natural **~3.9:1 class imbalance ratio**.
  - **Mitigation Strategy**: The pipeline resolves this imbalance through:
    1. Stratified patient-level splitting (preserving class ratios without slice leakage across splits).
    2. Inverse frequency class weighting in CrossEntropyLoss: $\text{Weight}_{\text{LGG}} = 2.45$, $\text{Weight}_{\text{HGG}} = 0.628$.
    3. Balanced 50/50 mini-batch sampling via PyTorch `WeightedRandomSampler`.
  - **Final Model Performance**: Using `swin_base_patch4_window7_224` (4-channel early fusion), the model achieves **86.73% test accuracy** and **0.9677 AUC** on the held-out 588-slice test split (112 LGG, 476 HGG) with 94.64% LGG recall and 84.87% HGG recall.
- **Metadata CSVs & Classification Labels**:
  - `name_mapping.csv`: Maps BraTS subject IDs across challenge years (2017–2020) and TCGA/TCIA subject IDs. **Crucially, the `Grade` column in this file provides the binary classification ground-truth label (`HGG` vs `LGG`)** for each subject ID (`BraTS20_Training_001` .. `369`), as training patient directories are organized flatly rather than in separate subfolders.
  - `survival_info.csv`: Contains clinical survival metadata (`Brats20ID`, `Age`, `Survival_days`, `Extent_of_Resection`).

### Data Quality & Anomaly Notes
- **`BraTS20_Training_355`**: Segmentation mask is named `W39_1998.09.19_Segm.nii` instead of `BraTS20_Training_355_seg.nii`. Ingestion scripts support this alias when loading segmentation volumes.

---

## 3. Preprocessing, Leakage Prevention & Skull-Stripping

- **Patient-Level Separation**: To eliminate data leakage, dataset splitting (`src/preprocessing/split_data.py`) partitions data strictly at the **patient volume level** rather than the slice level. Zero slices from any validation or test patient appear in the training split.
- **Skull-Stripping**: BraTS data is distributed pre-skull-stripped, so this step was bypassed for BraTS; the utility script ([`src/preprocessing/skull_strip.py`](file:///c:/PROJECTS/Explainable_Brain_Tumor_Diagnosis_Using_Vision_Transformers_and_Multi-Modal_MRI_Fusion/src/preprocessing/skull_strip.py)) is retained for raw or prospective external MRI scans.
- **Normalization**: Per-slice min-max intensity scaling to `[0.0, 1.0]` and bilinear interpolation to `[224, 224]` spatial resolution.


