# Run A: Swin-Tiny + BraTS Multi-Modal Fusion — Final Results Report

## Experiment Overview
- **Model:** Swin-Tiny (`swin_tiny_patch4_window7_224`)
- **Total Parameters:** **27,520,380** (~**27.52 Million**)
- **Dataset:** BraTS 2020 Multi-Modal MRI Fusion
- **Training Epochs:** 20
- **Best Validation Epoch:** Epoch 17
- **Best Validation Accuracy:** **87.93%**
- **Final Validation Accuracy (Epoch 20):** 87.76%
- **Verified Checkpoint Path:** `checkpoints/brats_tiny_best_model.pth`

## Held-Out Test Set Performance (Reported)
- **Test Accuracy:** **86.73%** (0.8673)
- **Test Precision (HGG):** 0.9662
- **Test Recall (HGG):** 0.9622 (458 / 476)
- **Test F1-Score (HGG):** 0.9642
- **Macro F1:** 0.7719
- **Test AUC (Binary):** **0.9555**
- **Class Recalls:** LGG (Minority): **85.71%** (96 / 112) | HGG (Majority): **96.22%** (458 / 476)
- **Status:** **FIXED** (Class-weighted loss & weighted random sampling successfully eliminated majority-class collapse, improving LGG recall from 0.0% to 85.71%).
