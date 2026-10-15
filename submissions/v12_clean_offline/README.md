# Submission Version 12: Cleaned Offline Baseline (Target LB: 0.956)

## Overview
- **Status:** **Offline Verified & Cleaned (Submission Locked)**
- **Forensic Fix Applied:** All 4 heuristic post-processing rules (`+0.15*ACL`, `+0.20*Effusion`, etc.) introduced in V9-V11 have been completely excised.
- **Blending Protocol:** Pure, mathematically optimal uniform 50/50 percentile rank blending across all 12 targets:
  $$r = 0.50 \cdot r_{\text{raptor}} + 0.50 \cdot r_{\text{convnext}}$$
- **Validation Receipts:**
  - `blend_receipt.json`: Verifies 50/50 weights across all classes, 0 study fallbacks, total run time ~438.5s.
  - `submission_receipt.json`: Validates shape `(N, 13)` and monotonicity across all prediction arms.
- **Rules Compliance:** Submission lock active until user explicitly authorizes live submission.
