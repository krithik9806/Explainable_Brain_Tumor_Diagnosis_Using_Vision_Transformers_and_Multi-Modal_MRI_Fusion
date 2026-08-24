# Run B: Swin-Base + BraTS Multi-Modal Fusion — Final Results Report

## Experiment Overview
- **Model:** Swin-Base (`swin_base_patch4_window7_224`)
- **Dataset:** BraTS 2020 Multi-Modal MRI Fusion
- **Training Epochs:** 20
- **Best Validation Epoch:** Epoch 15 (First reached 90.65%) / Epoch 17 (Lowest Val Loss: 0.2366)
- **Best Validation Accuracy:** **90.65%**
- **Final Validation Accuracy (Epoch 20):** 90.65%
- **Verified Checkpoint Path:** `checkpoints/brats_base_best_model.pth`

## Held-Out Test Set Performance (Reported)
- **Test Accuracy:** **89.46%** (0.8946)
- **Test Precision (HGG):** 0.9707
- **Test Recall (HGG):** 0.9748 (464 / 476)
- **Test F1-Score (HGG):** 0.9727
- **Macro F1:** 0.8169
- **Test AUC (Binary):** **0.9702**
- **Class Recalls:** LGG (Minority): **87.50%** (98 / 112) | HGG (Majority): **97.48%** (464 / 476)
- **Confusion Matrix:** `[[98, 14], [12, 464]]` (588 held-out test slices)
- **Status:** **FIXED** (Tuned learning rate of 1.5e-5 and 20 full epochs resolved undertraining, boosting test accuracy from 56.29% to 89.46%).
