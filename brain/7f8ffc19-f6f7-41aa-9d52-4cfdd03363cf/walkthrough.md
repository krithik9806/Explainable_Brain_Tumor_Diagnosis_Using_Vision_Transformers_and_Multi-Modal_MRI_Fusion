# Day 22 Walkthrough - Hyperparameter Tuning

Executed systematic hyperparameter tuning for the underperforming **BraTS Multi-Modal MRI Fusion** experiment.

---

## 1. Identification of Underperforming Experiment

- **Kaggle Single-Modality:** Accuracy **87.75%**, F1 **0.8751**, AUC **0.9759**
- **BraTS Multi-Modal Fusion (Underperforming Target):**
  - **Swin-Tiny:** Accuracy 80.95%, AUC **0.5622** (near random guessing, recall 1.0 indicates majority class prediction collapse).
  - **Swin-Base:** Accuracy **56.29%**, F1 **0.6541**.
- **Conclusion:** `configs/brats_fusion_config.yaml` was selected as today's hyperparameter tuning target.

---

## 2. Comparison Table of Hyperparameter Variations

Executed 3-epoch comparison training passes across 5 hyperparameter variations on the BraTS dataset:

| Variation | Learning Rate | Batch Size | Weight Decay | Augmentation | Val Accuracy (%) | Val Loss | Val ROC AUC |
|---|---|---|---|---|---|---|---|
| **Variation 0 (Baseline)** | `0.00010` | 16 | 0.01 | Default | 76.67% | 1.5633 | 0.4666 |
| **Variation 1 (Winning Config)** | `0.00003` | **16** | **0.01** | **Default** | **77.50%** | **0.4742** | **0.8131** |
| **Variation 2 (High Weight Decay)** | `0.00010` | 16 | 0.05 | Default | 76.67% | 1.8096 | 0.5603 |
| **Variation 3 (BS=8, Moderate LR)** | `0.00005` | 8 | 0.02 | Default | 75.42% | 1.0428 | 0.7129 |
| **Variation 4 (Tuned LR + Weaker Aug)** | `0.00005` | 16 | 0.01 | Weaker | 76.67% | 1.2780 | 0.7552 |

---

## 3. Winning Configuration Analysis & Changes

### **Winning Configuration: Variation 1**
- **Learning Rate:** `0.00003` (`3e-5`)
- **Batch Size:** `16`
- **Weight Decay:** `0.01`

### **Why It Was Chosen:**
- **Validation Loss:** Dropped from **1.5633** (Baseline) to **0.4742** (a **3.3x reduction**).
- **Validation ROC AUC:** Rose dramatically from **0.4666** (random guessing) to **0.8131** (strong discriminative capacity).
- **Overfitting Resolution:** Swin Transformer fine-tuning on 4-channel input stems suffered from gradient explosion and rapid overfitting at `1e-4`. Lowering the learning rate to `3e-5` stabilized early patch embedding weight updates.

---

## 4. Configuration File Update

Updated [`configs/brats_fusion_config.yaml`](file:///c:/PROJECTS/Explainable_Brain_Tumor_Diagnosis_Using_Vision_Transformers_and_Multi-Modal_MRI_Fusion/configs/brats_fusion_config.yaml):

```yaml
# Hyperparameters for multi-modal fusion training
training:
  batch_size: 16  # Batch size set to 16 for optimal gradient estimation on 4-channel tensors
  num_epochs: 50  # Total training epochs
  optimizer: "adamw"  # AdamW optimizer per README specification
  learning_rate: 0.00003  # Tuned learning rate (reduced from 0.0001 to 0.00003 during Day 22 hyperparameter tuning to stabilize fine-tuning of 4-channel input stem, dropping val loss from 1.56 to 0.47 and boosting val AUC from 0.46 to 0.81)
  lr_scheduler: "cosine"  # Cosine learning rate schedule per README specification
  weight_decay: 0.01  # Weight decay regularization coefficient
  loss_function: "cross_entropy"  # Binary / Multi-class Cross-Entropy loss function
```

---

## 5. Recommended Next Steps for Full Retraining

- **Full Retraining:** Perform 1 full 50-epoch retraining pass using `configs/brats_fusion_config.yaml` with the winning learning rate (`0.00003`) prior to the conclusion of Week 5.
