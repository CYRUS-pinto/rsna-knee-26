# V13 Tri-Hybrid Aligned Rank Fusion Submission

## Overview
V13 is our primary candidate submission designed to break past the **0.956** public leaderboard clump into **Gold Tier (0.963+)** while strictly adhering to the **$18,000 Efficiency Prize** latency constraint ($T \le 500$s).

## Core Architecture
V13 merges three fundamentally orthogonal inference engines via scale-invariant uniform 1/3 percentile rank averaging:
1. **Engine 1: Raptor CoAtNet-384 SWA** (Global hybrid transformer, 44 2.5D temporal windows, ACL/Contusion specialist).
2. **Engine 2: Heliosli ConvNeXt-S6 R7** (Pure modern ConvNet, 4-fold 352x352, Osteoarthritis Extent heads).
3. **Engine 3: Multi-Plane Slice-Transformer** (Sequence-aware multi-plane attention, Sagittal/Coronal/Axial plane embeddings, PF OA & Baker's Cyst specialist).

## Mathematical Formulation
For each of the $C = 12$ clinical abnormalities across $N$ test studies:
$$r_1 = \frac{\text{argsort}(\text{argsort}(p_{\text{coatnet}}))}{N}, \quad r_2 = \frac{\text{argsort}(\text{argsort}(p_{\text{convnext}}))}{N}, \quad r_3 = \frac{\text{argsort}(\text{argsort}(p_{\text{multiplane}}))}{N}$$

$$\mathbf{R}_{\text{final}} = \frac{1}{3} r_1 + \frac{1}{3} r_2 + \frac{1}{3} r_3$$

- **Scale-Invariant**: Completely neutralizes heterogeneous logit scales and calibration shifts between models.
- **Zero Heuristics**: Adheres strictly to [ADR-002](../../decisions/ADR-002-excising-heuristic-priors-and-overfitting.md); no additive prior distortion.
- **Monotonic**: Preserves ROC-AUC ordering without threshold tuning.

## Efficiency Profile
- Rank Fusion execution time: **0.000023s per study** (0.030s for the entire hidden test set).
- Total pipeline latency: **~416s** on Kaggle hidden test set ($84$ seconds of safe buffer under the $500$s limit).

## Key Files
- [`submission.py`](submission.py): Production submission script with rank fusion and non-raising runtime correlation guard (`min_rho >= 0.30`).
- [`submission.md`](submission.md): Comprehensive architectural, algorithmic, and benchmark documentation.
