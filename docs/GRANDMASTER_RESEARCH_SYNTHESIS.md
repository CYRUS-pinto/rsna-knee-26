# Grandmaster Research & Strategy Synthesis: Breaking the 0.957 Frontier

## 1. Executive Summary: What Separates Top 10 (0.965) from the Plateau (0.956)
A rigorous audit of the latest Kaggle research literature (including Suzuki Yuto's empirical ablation `knee-0-939-to-0-956-what-moved-the-score`, nartaa's `rsna-knee-s6fall-solo`, and Heliosli's multi-scale architecture) reveals the definitive competitive truths:

1. **Re-weighting & Post-Processing Alone Plateaus:**
   - Pure weight grid-searching or heuristic post-processing (e.g. cross-finding re-ranking of Synovitis with Effusion) fails to generalize and empirically degraded validation AUC by up to **-0.020**.
   - Every major step on the leaderboard came from **newly trained, orthogonal model representations**, not re-weighting existing predictions.

2. **The Exact Steps That Moved the Metric:**
   | Progression | Core Architectural Shift | Gain |
   | :--- | :--- | :--- |
   | **0.924 → 0.941** | Dreaddevelopment Raptor baseline (single-fold CoAtNet 384) | Baseline |
   | **0.941 → 0.945** | Multi-plane attention MIL & Depth-Zone readers | +0.004 |
   | **0.945 → 0.950** | Nartaa's CoAtNet-384 SWA (trained with soft report labels) | +0.005 |
   | **0.950 → 0.956** | Heliosli's R7 ConvNeXt-Small 4-fold ensemble + dedicated fluid stackers | +0.006 |
   | **0.956 → 0.957** | Spatial Multi-Scale Fusion: 384px full-field + 224px center-crop CoAtNet + R7 | +0.001 |
   | **0.957 → 0.960+ (Active)** | 4-Model Multi-Scale Quad-Ensemble + Anatomical Flip TTA | In Queue |

---

## 2. Active Kaggle Evaluation Queue
We currently have two high-tier candidates running directly on Kaggle's hidden evaluation test set:

1. **Submission `57237329` (V14 Multi-Scale 0.957 Baseline):**
   - Dual T4 runtime: 476.3s.
   - Ensembles 384px CoAtNet + 224px CoAtNet + ConvNeXt R7.
   - Status: `SubmissionStatus.PENDING`.

2. **Submission `57238387` (V15 Multi-Scale with Anatomical Flip TTA):**
   - Dual T4 runtime: 431.0s.
   - Enables bilateral reflection TTA on Coronal and Axial planes with anatomical compartment permutation (Medial $\leftrightarrow$ Lateral).
   - Status: `SubmissionStatus.PENDING`.

---

## 3. The Grandmaster Quad-Ensemble Blueprint (V16)
To break past 0.957 towards 0.960+, we integrate the newly verified **S6-Fall ConvNeXt** (`nartaa/rsna-knee-s6fall-weights`, 0.950 solo) with Heliosli R7 and the dual-scale CoAtNet:

$$\text{Final Rank} = 0.40 \times \text{Rank}(\text{ConvNeXt R7}) + 0.20 \times \text{Rank}(\text{ConvNeXt S6-Fall}) + 0.40 \times \text{Rank}(\text{Raptor Multi-Scale})$$

Where:
$$\text{Rank}(\text{Raptor Multi-Scale}) = 0.65 \times \text{Rank}(\text{CoAtNet-384}) + 0.35 \times \text{Rank}(\text{CoAtNet-224})$$

### Architectural Diversity Matrix:
- **Arm 1 (CoAtNet-384):** Hybrid self-attention + convolutions at 384px resolution. Captures fine-grained focal tears (ACL, MCL).
- **Arm 2 (CoAtNet-224):** 0.8 center-crop resized to 224px. Captures global joint geometry and effusions.
- **Arm 3 (ConvNeXt-Small R7):** 4-fold pure convolutional depthwise architecture with specialized volumetric stackers for Effusion & Baker's Cyst.
- **Arm 4 (ConvNeXt-Small S6-Fall):** Independent seed and fold training by nartaa across 4,263 studies using JEV-calibrated continuous report labels.
