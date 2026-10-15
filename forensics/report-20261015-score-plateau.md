# GSD Forensic Investigation: RSNA Knee Score Plateau & Root-Cause Diagnosis

**Date:** 2026-10-15  
**Target:** RSNA Knee Abnormality Detection  
**Lead Agent:** Antigravity (Advanced Agentic Systems)  
**Investigation Scope:** Why Submissions V7, V9, V10, V11 Plateaud at 0.955 vs Personal Best V6 (0.956)

---

## 1. Executive Forensic Verdict

Our personal best submission remains **Version 6 at 0.956**.  
Subsequent iterations (V7, V9, V10, V11) all plateaued at **0.955** (-0.001 regression).

### The Root Cause in One Sentence:
**We optimized asymmetric weights and heuristic priors against the 58 Gold studies (an under-sampled $N=58$ set with as few as 9 positive cases), which overfit local noise, disrupted natural test rank orderings, and diluted the mathematically optimal uniform rank fusion of Version 6.**

---

## 2. Forensic Timeline & Score Matrix

| Submission | Configuration | Offline Gold-58 | Public LB | Delta vs PB | Root Cause Analysis |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Version 6** | **Raptor CoAtNet (50%) + Heliosli ConvNeXt R7 (50%)** | 0.9160 | **`0.956`** | **BASELINE (PB)** | **Pure, unadulterated rank fusion.** Minimized estimator variance across two strong, orthogonal model families. |
| **Version 7** | Asymmetric Anatomical Weights (CoAtNet 65% tears, ConvNeXt 70% OA) | 0.9240 (+0.008) | `0.955` | -0.001 | **Gold-58 Overfitting.** Moving away from 50/50 reduced ensemble variance reduction on the larger unseen test set. |
| **Version 9** | Tri-Engine: Raptor + Cyrus Colab SWA 0.9167 + ConvNeXt R7 | 0.9280 (+0.012) | `0.955` | -0.001 | **Architecture Redundancy.** Cyrus SWA was another CoAtNet trained on $k=12$ slices, adding correlated variance rather than orthogonal signal. |
| **Version 10** | Tri-Engine + 3 Clinical Physiological Priors (Synovitis+Eff, Contusion+ACL, Baker+Eff) | 0.9324 (+0.016) | `0.955` | -0.001 | **Rank Inversion Penalty.** Adding `+0.15*ACL` to Contusion promoted false positives on patients with isolated tears, degrading individual class AUC. |
| **Version 11** | Tri-Engine + 4 Clinical Priors + Aggressive Asymmetric Arm Weights | 0.9331 (+0.017) | `0.955` | -0.001 | **Compound Distortion.** Compounded asymmetric weight skew and prior leakage. Gold AUC showed false peak due to tiny positive counts (MCL=9). |

---

## 3. Mathematical Mechanism: Why Priors & Asymmetry Failed on LB

### A. The Rank-Order Sensitivity of ROC AUC
ROC AUC is the Wilcoxon-Mann-Whitney rank statistic:
$$\text{AUC} = \frac{1}{N_+ N_-} \sum_{i \in \text{Pos}} \sum_{j \in \text{Neg}} \mathbb{I}(s_i > s_j)$$

Notice:
* AUC cares **only** about pairwise ranking ($s_i > s_j$).
* When we added heuristic adjustments such as:
  $$\text{Contusion}_{\text{adj}} = \text{Contusion} + 0.15 \times \text{ACL}$$
  Consider two test patients:
  * Patient 1: True Contusion (+), No ACL Tear (-) $\rightarrow$ Contusion logit = 0.70, ACL = 0.10 $\rightarrow \text{Score} = 0.715$
  * Patient 2: No Contusion (-), Severe ACL Tear (+) $\rightarrow$ Contusion logit = 0.65, ACL = 0.95 $\rightarrow \text{Score} = 0.7925$
* The prior promoted Patient 2 above Patient 1! This inverted the rank order for Contusion, directly degrading Contusion AUC.
* Because Macro-AUC weights all 12 classes equally ($1/12$), losing 0.015 AUC on Contusion or Fracture pulls down the entire macro average by >0.001.

### B. The Law of Small Numbers on the 58 Gold Studies
In `train.csv`, the 58 Gold adjudicated studies contain:
* **MCL:** Only 9 positive cases.
* **Medial OA:** Only 15 positive cases.
* **Lateral OA:** Only 11 positive cases.
* **Fracture:** Only 7 positive cases.

Flipping the ranking of just **one single case** in Fracture changes the Fracture AUC on Gold-58 by:
$$\Delta \text{AUC} = \frac{1}{7 \times 51} \approx 0.0028$$
Optimizing per-finding weights or heuristic multipliers on Gold-58 was fitting to stochastic noise rather than true population statistics.

---

## 4. What The 0.960+ Competitors Are ACTUALLY Doing

Forum mining across 186 topics and 1,094 messages reveals the exact architectural differences between the 0.954-0.956 cluster and the >0.960 leaders:

| Dimension | Our Current Setup | >0.960 Grandmaster Setup |
| :--- | :--- | :--- |
| **Model Families** | CoAtNet (Hybrid) + ConvNeXt (CNN) | **CoAtNet + ConvNeXt + DINOv2 (Self-Supervised ViT)**. DINOv2 provides totally orthogonal patch tokens. |
| **Slice Coverage** | 12 to 24 slices sampled via central heuristic | **80 to 96 slices per study** across Sagittal, Coronal, and Axial planes. |
| **Physical Voxel Geometry** | Resizing arbitrary slices to 384px | **Physical mm Normalization:** 130mm - 140mm bounding box centered on joint line, projected along normal vector $\vec{n} = \text{IOP}_{0:3} \times \text{IOP}_{3:6}$. |
| **Supervision Strategy** | Single pseudo-label set (Super-Consensus) | **Iterative Multi-Teacher Pseudo-Labeling:** Train Vision Model $\rightarrow$ generate out-of-fold test predictions $\rightarrow$ blend with LLM reports $\rightarrow$ retrain. |
| **Ensembling Math** | Manual weights, heuristic priors | **Strict Uniform Rank Percentile Fusion:** No hand-crafted post-processing rules. Equal weighting of diverse model families. |

---

## 5. Corrective Action Plan (How We Exceed 0.956)

1. **Restore Baseline Cleanliness:**
   Excise all heuristic priors (`Synovitis+Effusion`, `Contusion+ACL`, etc.) from inference. Keep rank blending strictly clean and uncorrupted.
2. **True Model Family Diversity (Engine 3):**
   Do not add another CoAtNet. Train a **DINOv2-Small/Base** or **3D Slice-Transformer** that learns features CoAtNet cannot see.
3. **Multi-Scale Depth:**
   Increase slice sampling from 12 to 24+ slices to capture radial and meniscus root tears.
4. **Constraint Enforced:**
   No submissions will be dispatched without explicit user command.
