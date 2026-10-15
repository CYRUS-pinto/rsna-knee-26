# Submission Version 12: Cleaned Offline Baseline

## Overview
- **Status:** **Offline Verified Pipeline** (Local notebook prepared with priors removed).
- **Context on Kaggle vs Local:**
  - On Kaggle, an earlier running commit titled "V12" was launched with "+ Priors" (Super-Consensus V2 + Priors).
  - This local directory contains the version where Cell 6 has all heuristic priors excised, restoring pure 50/50 uniform rank blending.
- **Blending Protocol:** Uniform 50/50 percentile rank blending across all 12 targets:
  $$r = 0.50 \cdot r_{\text{raptor}} + 0.50 \cdot r_{\text{convnext}}$$
- **Validation Receipts:**
  - `blend_receipt.json`: 50/50 weights across all classes, 0 study fallbacks, commit time ~438.5s.
  - `submission_receipt.json`: Validates shape `(N, 13)` and monotonicity across all prediction arms.
- **Decision:** Ready as an unadjusted candidate once the Kaggle running version completes its evaluation.
