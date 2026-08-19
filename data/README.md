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
- **Metadata CSVs & Classification Labels**:
  - `name_mapping.csv`: Maps BraTS subject IDs across challenge years (2017–2020) and TCGA/TCIA subject IDs. **Crucially, the `Grade` column in this file provides the binary classification ground-truth label (`HGG` vs `LGG`)** for each subject ID (`BraTS20_Training_001` .. `369`), as training patient directories are organized flatly rather than in separate subfolders.
  - `survival_info.csv`: Contains clinical survival metadata (`Brats20ID`, `Age`, `Survival_days`, `Extent_of_Resection`).

### Data Quality & Anomaly Notes
- **`BraTS20_Training_355`**: Segmentation mask is named `W39_1998.09.19_Segm.nii` instead of `BraTS20_Training_355_seg.nii`. Ingestion scripts must support this alias when loading segmentation volumes.

---

## 3. Preprocessing & Skull-Stripping Note

> [!NOTE]
> BraTS data is distributed pre-skull-stripped, so this step was skipped for the current pipeline; the function (`src/preprocessing/skull_strip.py`) exists for reusability if raw/new MRI data is added later.


