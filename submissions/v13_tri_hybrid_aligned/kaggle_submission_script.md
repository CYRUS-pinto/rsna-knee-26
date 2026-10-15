# `submissions/v13_tri_hybrid_aligned/kaggle_submission_script.py`

## Purpose
This script is the production Kaggle submission kernel for V13 Tri-Hybrid Aligned, engineered to execute offline on the hidden test set within the competition's runtime limit ($T \le 500$s) to claim the **$18,000 Efficiency Prize** while maximizing Macro-AUC via Grandmaster Specialist Column Routing.

## Key Architectural Highlights

### 1. Sequential GPU Execution & Memory Flushing
To prevent CUDA out-of-memory crashes on single-GPU Kaggle instances:
- Engine 1 (CoAtNet-384 Raptor) runs to completion $\to$ `del model; gc.collect(); torch.cuda.empty_cache()`
- Engine 2 (ConvNeXt-S6 R7) runs to completion $\to$ `del model; gc.collect(); torch.cuda.empty_cache()`
- Engine 3 (Multi-Plane Slice-Transformer) runs to completion $\to$ `del model; gc.collect(); torch.cuda.empty_cache()`

### 2. Patient-Coordinate Laterality Derivation
Calculates $p_{x, \text{center}}$ from `ImagePositionPatient[0]` to canonically identify Left vs Right knee, protecting the 5 compartment-dependent targets (`MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`).

### 3. Grandmaster Specialist Column Routing (ADR-013)
- **4 Focal Pathology Columns (`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`):**
  $$\mathbf{S}_{\text{focal}} = 0.40 \cdot \text{Rank}(E_1) + 0.20 \cdot \text{Rank}(E_2) + 0.40 \cdot \text{Rank}(E_3)$$
- **8 Systemic & Fluid Columns (`Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`):**
  $$\mathbf{S}_{\text{systemic}} = 0.25 \cdot \text{Rank}(E_1) + 0.50 \cdot \text{Rank}(E_2) + 0.25 \cdot \text{Rank}(E_3)$$

### 4. Non-Raising Correlation Safeguard
Monitors inter-arm correlation $\rho$. If $\rho < 0.30$, it safely falls back to uniform 1/3 rank averaging without raising runtime exceptions or terminating the evaluation.

## Verification
Run standalone verification locally:
```bash
python submissions/v13_tri_hybrid_aligned/kaggle_submission_script.py
```
Outputs `submission.csv` and verifies row formatting, NaN absence, and bounds $[0.0, 1.0]$.
