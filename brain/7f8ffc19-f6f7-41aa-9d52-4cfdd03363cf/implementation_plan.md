# Implementation Plan - Day 22: Hyperparameter Tuning for Brain Tumor Diagnosis Pipeline

## Goal Description
Perform systematic hyperparameter tuning for the underperforming experiment identified from Day 21's results (BraTS Multi-Modal MRI Fusion). Run short comparison training passes (5 epochs each) across 3–5 hyperparameter variations (tuning learning rate, batch size, weight decay, and augmentation strength), log metrics to Weights & Biases (W&B), build a comparative evaluation table, and update the winning configuration into `configs/brats_fusion_config.yaml`.

---

## 1. Identification of Underperforming Experiment

Based on Day 21's evaluation results reported in `results/` and the `README.md` Results table:

| Model | Dataset / Modality | Accuracy | Precision | Recall | F1-score | AUC |
|---|---|---|---|---|---|---|
| Swin-Tiny | Kaggle Single-Modality | **87.75%** | 0.8908 | 0.8775 | **0.8751** | **0.9759** |
| Swin-Tiny | BraTS Multi-Modal Fusion | **80.95%** | 0.8095 | 1.0000 | 0.8947 | **0.5622** |
| Swin-Base | BraTS Multi-Modal Fusion | **56.29%** | 0.9101 | 0.5105 | 0.6541 | **0.7315** |

### **Target Experiment Selected for Tuning: BraTS Multi-Modal MRI Fusion (`configs/brats_fusion_config.yaml`)**

**Rationale:**
- **Kaggle single-modality** achieves high performance across all metrics (87.75% Accuracy, 0.9759 AUC).
- **BraTS multi-modal fusion** demonstrates severe performance degradation:
  - **Swin-Tiny** gets an AUC of **0.5622** (barely above random chance 0.50), with a Recall of 1.0000 indicating class imbalance collapse (predicting the majority class).
  - **Swin-Base** gets only **56.29% accuracy** and 0.6541 F1-score, suffering from optimization difficulty and potential overfitting on 4-channel MRI tensors.
- Therefore, `configs/brats_fusion_config.yaml` is the clear target for Day 22 hyperparameter tuning.

---

## 2. Proposed Hyperparameter Variations (Short 5-Epoch Comparison Runs)

We will test 4 distinct hyperparameter variations for the BraTS fusion experiment:

1. **Variation 0 (Baseline / Current Config):**
   - Learning Rate: `0.0001` (1e-4)
   - Batch Size: `16`
   - Weight Decay: `0.01`
   - Augmentation: Default (`flip_prob=0.5`, `rotation=15`, `elastic=True`)

2. **Variation 1 (Lower Learning Rate - Fine-Tuning Focus):**
   - Learning Rate: `0.00003` (3e-5)
   - Batch Size: `16`
   - Weight Decay: `0.01`
   - Rationale: Swin Transformer fine-tuning with 4-channel input stems can suffer from gradient instability at 1e-4. Lower LR provides smoother convergence.

3. **Variation 2 (Higher Regularization & Weight Decay):**
   - Learning Rate: `0.0001` (1e-4)
   - Batch Size: `16`
   - Weight Decay: `0.05` (5x increase)
   - Rationale: Combats potential overfitting and stabilizes attention weight updates.

4. **Variation 3 (Adjusted Batch Size & Moderate LR):**
   - Learning Rate: `0.00005` (5e-5)
   - Batch Size: `8` (Smaller batch size for higher gradient noise and finer optimization trajectory)
   - Weight Decay: `0.02`

5. **Variation 4 (Tuned Learning Rate & Reduced Augmentation Noise):**
   - Learning Rate: `0.00005` (5e-5)
   - Batch Size: `16`
   - Weight Decay: `0.01`
   - Augmentation: Moderate (`flip_prob=0.3`, `rotation=10`)

---

## User Review Required

> [!NOTE]
> The hyperparameter tuning runs will be executed for 5 epochs each to quickly compare directional improvement in validation loss, validation accuracy, and ROC AUC without requiring long compute cycles.

> [!IMPORTANT]
> The winning hyperparameter set will be saved directly into `configs/brats_fusion_config.yaml` with clear inline documentation comments explaining the rationale for the change.

---

## Proposed Changes

### Configuration Layer

#### [MODIFY] [brats_fusion_config.yaml](file:///c:/PROJECTS/Explainable_Brain_Tumor_Diagnosis_Using_Vision_Transformers_and_Multi-Modal_MRI_Fusion/configs/brats_fusion_config.yaml)
- Update `training.learning_rate`, `training.batch_size`, `training.weight_decay`, and/or `augmentation` settings to match the winning hyperparameter variation.
- Add explanatory comments detailing the Day 22 tuning findings and why the new values were selected.

---

## Verification Plan

### Automated Verification
1. Run each 5-epoch variation using `python src/train.py --config configs/brats_fusion_config.yaml ...` or a python execution wrapper.
2. Verify W&B run logging for loss, accuracy, and AUC metrics.
3. Compare validation metrics across all 4-5 variations in a comparison table:
   `Variation | Learning Rate | Batch Size | Weight Decay | Val Accuracy | Val Loss | Val AUC`
4. Confirm `configs/brats_fusion_config.yaml` contains valid YAML syntax by loading it with PyYAML.

### Manual Verification
- Review the final comparison table and verify that the best config selection is logically justified by empirical metrics.
