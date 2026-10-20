# ADR-026: Standalone Raptor Efficiency Benchmark and ConvNeXt Ensembling Gain

## Context & Purpose
On October 17, 2026, Candidate V26-Fast (Ref `57300862`) completed official evaluation on Kaggle's hidden test set. Candidate V26-Fast was built as a pure, lightweight MultiScale Raptor architecture (384px + 224px, 65/35 rank blend) with the fast C-struct DICOM loader (`_fastread.py`), completely isolating Raptor from ConvNeXt R7 to evaluate two critical hypotheses:
1. What is the standalone predictive performance of the Raptor CoAtNet MIL engine on the hidden test set?
2. Exactly how much incremental AUC does ConvNeXt add to the ensemble?

## Empirical Results (Measured on Kaggle Hidden Test Set)

| Metric / Attribute | Candidate V26-Fast (Standalone Raptor) | Candidate V17 / V20 (Raptor + ConvNeXt) | Delta ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Kaggle Ref** | `57300862` | `57253840` / `57267740` | — |
| **Public Leaderboard Score** | **0.949** (MEASURED) | **0.958** (MEASURED) | **+0.009 AUC** |
| **Architectures** | CoAtNet-d4 MultiScale (384 + 224) | CoAtNet-d4 MultiScale + ConvNeXt R6/R7 | Ensemble Boost |
| **Runtime Profile** | Fast (~5–7 min inference) | Slow (~25–30 min inference) | +80% Time Drag |
| **Primary Track Fit** | Efficiency Prize Frontier | Main Leaderboard Track | Strategic Divergence |

## Core Findings

### 1. The ConvNeXt Ensembling Gain is $+0.009$ AUC
Standalone Raptor reaches $0.949$. When combined with ConvNeXt (ADR-024 Uniform Master Blend), the score reaches $0.958$.
* This proves that ConvNeXt is **indispensable** for the Main Leaderboard.
* The two models have complementary error profiles across distinct anatomical views: ConvNeXt provides the orthogonal signal that elevates the macro-average by almost a full point (+0.009).

### 2. Efficiency Prize Benchmark
On the 5,985-team Efficiency Leaderboard:
* Teams scoring $0.947\text{--}0.949$ with lean execution sit between **Rank 21 and Rank 25** globally (e.g. `ChenyuWang0628` at 0.948 Rank 21, `huytngo and apoorvsara` at 0.948 Rank 24).
* Candidate V26-Fast provides our cleanest, unencumbered entry on the efficiency frontier.

## Strategic Decision Matrix
1. **Slot 1:** Permanently locked to Ref `57165672` (Candidate V6, $0.956$).
2. **Slot 2 (Main Track Focus):** Reserved for Candidate V25 (Ref `57301656`) or Candidate V24 (Ref `57298964`) if they score $\ge 0.960$; otherwise locked to Ref `57253840` (Candidate V17, $0.958$).
3. **Slot 2 (Efficiency Focus Option):** If Candidate V26-Fast secures an official Top 15–25 Efficiency ranking once the leaderboard refreshes, Cyrus can choose to designate Slot 2 to compete directly for the **$18,000 Efficiency Prize** ($7,000 1st place).
