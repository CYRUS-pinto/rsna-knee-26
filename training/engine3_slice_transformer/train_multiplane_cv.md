# Multi-Plane 5-Fold Cross-Validation Trainer (`train_multiplane_cv.py`)

## 1. Overview
`train_multiplane_cv.py` is the full 5-fold cross-validation training orchestrator for Engine 3. Unlike single-fold quick trainers, this script manages out-of-fold predictions, cross-entropy loss tracking, and stochastic weight averaging across all 5 folds to generate full-dataset out-of-fold predictions.

---

## 2. Key Components

### 2.1 Study Dataset with Anatomy-Guided Windowing
The dataset loads volume arrays (`all_vols.npy`, `extra_vols.npy`) and valid slice segmentation masks (`all_masks.npy`, `extra_masks.npy`).
- For each study, the script identifies the active anatomical range $[c_{\min}, c_{\max}]$ using the segmentation masks.
- Non-informative empty background slices at the volume edges are discarded.
- Slabs of 3 contiguous slices are constructed: $\text{Slab}_c = [c-1, c, c+1]$ at normalized $[0, 1]$ intensity.

### 2.2 5-Fold Loop & Out-of-Fold Matrix
The trainer iterates across folds $k \in \{0, 1, 2, 3, 4\}$:
1. Trains on all studies where $\text{fold} \ne k$.
2. Validates on fold $k$.
3. Stores raw logit predictions in an out-of-fold prediction matrix $\mathbf{L}_{\text{OOF}} \in \mathbb{R}^{4407 \times 12}$.
4. Upon loop completion, evaluates the global out-of-fold ROC-AUC across all 4,407 studies.

---

## 3. Training Safeguards & Convergence Checks
- **Differential Learning Rates**: Lower learning rate for pre-trained backbone ($3 \times 10^{-5}$) to prevent catastrophic forgetting; higher learning rate for the newly initialized Multi-Plane Transformer Head ($2 \times 10^{-4}$).
- **Gradient Clipping**: Maximum norm of 1.0 applied before each optimizer step.
- **Logit Range Sanity Checks**: Monitored after every validation epoch to verify predictions do not saturate in extreme negative or positive regimes.
