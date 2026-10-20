# ADR-028: Calibrated Grandmaster Resolution & Candidate V28 Deployment

**Status:** ACCEPTED  
**Date:** 2026-10-17  
**Author:** Cyrus Pinto & Pair Engineering  
**Context:** RSNA Knee Abnormality Detection (5 days remaining)

---

## 1. Context & Empirical Post-Mortem of Candidate V24

On October 17, 2026 at 06:25 UTC, Candidate V24 (Ref `57298964`) was submitted to Kaggle. Evaluation completed at 15:36 local time, scoring:
$$\text{Public LB: } \mathbf{0.956} \quad (\Delta = -0.002 \text{ vs PB } 0.958)$$

### Root Cause Analysis:
1. **The Flawed Hypothesis:** ADR-024 observed high Spearman correlation ($\rho > 0.82$) on Medial/Lateral compartments in the 120-study audit and incorrectly generalized that a uniform $0.50 / 0.50$ blend across all 12 pathologies was optimal.
2. **Pathology Inductive Bias Breakdown:**
   - **Synovitis:** Requires whole-volume spatial attention across the entire knee joint cavity. Raptor CoAtNet-384 specializes in this. In V17 ($0.958$), Raptor had $0.80$ weight. In V24 ($0.956$), forcing $0.50$ gave ConvNeXt too much authority, degrading Synovitis predictions.
   - **Lateral OA:** Requires tight 384px spatial resolution on joint-space narrowing. ConvNeXt R7 specializes in this. In V17 ($0.958$), ConvNeXt had $0.80$ authority (Raptor $0.20$). In V24 ($0.956$), forcing $0.50$ weakened this specialized signal.
   - **Baker's Cyst:** Fluid stackers in ConvNeXt provide strong signal ($0.65$ authority in V17 vs $0.50$ in V24).
   - **Absence of Flip TTA:** Candidate V24 was executed without horizontal flip TTA on ConvNeXt.

---

## 2. Decision: Candidate V28 (The Calibrated Grandmaster Master)

To resolve the regression and establish our definitive master pipeline:
1. **Directory:** `submissions/v28_calibrated_grandmaster_master`
2. **Notebook:** `rsna-knee-v28-calibrated-grandmaster-master.ipynb`
3. **Restored & Calibrated Specialist Weights:**
   ```python
   W_RAPTOR_C = {
       'Lateral OA': 0.20,     # ConvNeXt R7 80% authority
       'Synovitis': 0.80,      # Raptor 80% authority
       'PF OA': 0.60,          # Raptor 60% authority
       "Baker's": 0.35,        # ConvNeXt 65% authority
       'ACL': 0.60,            # Raptor 60% authority
       'MCL': 0.50,            # Balanced
       'Medial Meniscus': 0.50,# Balanced
       'Lateral Meniscus': 0.50,# Balanced
       'Medial OA': 0.50,      # Balanced
       'Effusion': 0.50,       # Balanced
       'Contusion': 0.50,      # Balanced
       'Fracture': 0.50        # Balanced
   }
   ```
4. **ConvNeXt R7 Flip TTA:** Re-enabled (`FLIP_TTA = True`).
5. **MultiScale Raptor Calibration:** $70\%$ 384px SWA and $30\%$ 224px SWA (`A224 = 0.30`).
6. **Kernel Push & Commit:** Kernel commit completed with status `COMPLETE` (0 errors).
7. **Official Kaggle Submission:** Dispatched under Ref **`57308728`** at 11:09:01 UTC.

---

## 3. Quota Status & Daily Accounting

- **Submissions Used Today:** 5 of 5 (100% daily allocation utilized).
- **Submissions Status:**
  - Ref `57293215`: Candidate V23 -> `0.957`
  - Ref `57298964`: Candidate V24 -> `0.956`
  - Ref `57300862`: Candidate V26-Fast -> `0.949` (~5 min, Efficiency Frontier)
  - Ref `57301656`: Candidate V25 -> `PENDING`
  - Ref `57308728`: Candidate V28 -> `PENDING` (Official Grandmaster Fix)
