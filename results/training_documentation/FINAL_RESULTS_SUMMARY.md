# Final Results Summary — Vision Transformer Multi-Modal Brain Tumor Fusion

This document summarizes the reported training and evaluation performance across all primary model experiments in the project.

## Primary Experiment Benchmark Table

| Model | Dataset | Epochs | Best Val Epoch | Best Val Accuracy | Test Accuracy | Test F1 | Test AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Swin-Tiny | BraTS Multi-Modal | 20 | 17 | 87.93% | 86.73% | 0.9642 | 0.9555 |
| Swin-Base | BraTS Multi-Modal | 20 | 15 | 90.65% | 89.46% | 0.9727 | 0.9702 |
| Swin-Tiny | Kaggle 4-Class | — | — | — | 87.75% | 0.8751 | 0.9759 |

## Key Result
> **KEY RESULT:**
> Swin-Base achieved the highest test accuracy among these reported experiments at **89.46%**, while achieving a best validation accuracy of **90.65%** and a test AUC of **0.9702**.

## Important Interpretation & Methodology Notes
1. **Validation vs. Test Distinction:**
   - **Best Validation Accuracy** (90.65% for Swin-Base, 87.93% for Swin-Tiny) is evaluated on the validation split during training for model selection.
   - **Test Accuracy** (89.46% for Swin-Base, 86.73% for Swin-Tiny) is evaluated strictly on the un-seen held-out test split (588 slices / 140 patient split).
   - Do not claim 90.65% test accuracy. The correct reported test accuracy for Swin-Base is **89.46%**.

2. **Class-Imbalance Resolution:**
   - **Previous Run Issue:** Initial unweighted Swin-Tiny training collapsed to majority-class prediction (100% HGG predictions, 0% LGG recall, 80.95% naive accuracy, 0.5185 AUC).
   - **Solution Implemented:** Class-weighted cross-entropy loss ($	ext{Weight}_{\text{LGG}} = 2.45$, $	ext{Weight}_{\text{HGG}} = 0.628$) combined with `WeightedRandomSampler` for 50/50 mini-batch sampling.
   - **Outcome:** LGG minority recall increased to **85.71%** (Swin-Tiny) and **87.50%** (Swin-Base), with AUC reaching **0.9555** and **0.9702** respectively.

3. **Swin-Base Undertraining Resolution:**
   - **Previous Run Issue:** Swin-Base previously completed only 1 epoch, yielding an uncalibrated accuracy of 56.29%.
   - **Solution Implemented:** Full 20-epoch training with a reduced learning rate ($1.5 \times 10^{-5}$) appropriate for fine-tuning the 86.7M parameter backbone.
   - **Outcome:** Test accuracy improved from 56.29% to **89.46%**, establishing Swin-Base as the top-performing model.
