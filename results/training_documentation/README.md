# Training & Evaluation Documentation Directory

This directory contains permanent, mentor-ready archival documentation of all model training and evaluation results produced in this project.

> **NOTE:** These files archive the reported training/evaluation results for mentor review. All values, epoch logs, and metrics reflect the actual reported outputs generated during training and evaluation sessions.

## Directory Structure

```text
results/training_documentation/
├── README.md                           # Overview and directory guide
├── FINAL_RESULTS_SUMMARY.md            # Primary mentor-ready benchmark summary
├── RUN_A_SWIN_TINY_BRATS/              # Swin-Tiny + BraTS 4-Channel Fusion (20 Epochs)
│   ├── epoch_results.csv               # Complete 20-epoch CSV dataset
│   ├── epoch_results.md                # Markdown epoch table & metadata
│   ├── training_log.txt                # Plain-text archival console log
│   ├── final_results.md                # Final test evaluation metrics
│   └── training_curves.png             # Training vs Val Loss and Accuracy plots
└── RUN_B_SWIN_BASE_BRATS/              # Swin-Base + BraTS 4-Channel Fusion (20 Epochs)
    ├── epoch_results.csv               # Complete 20-epoch CSV dataset
    ├── epoch_results.md                # Markdown epoch table & metadata
    ├── training_log.txt                # Plain-text archival console log
    ├── final_results.md                # Final test evaluation metrics
    └── training_curves.png             # Training vs Val Loss and Accuracy plots
```

## Recommended Mentor Review File
Open [`FINAL_RESULTS_SUMMARY.md`](file:///c:/PROJECTS/Explainable_Brain_Tumor_Diagnosis_Using_Vision_Transformers_and_Multi-Modal_MRI_Fusion/results/training_documentation/FINAL_RESULTS_SUMMARY.md) to review the final comparative summary.
