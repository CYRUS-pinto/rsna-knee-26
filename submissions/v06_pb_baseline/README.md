# Submission Version 6: Personal Best Baseline (Public LB: 0.956)

## Overview
- **Kaggle Ref:** `57165672`
- **Public LB Score:** **0.956** (Personal Best)
- **Engine Configuration:** Dual-Engine Parallel Inference
  - **Engine 1:** CoAtNet-384 SWA (`dreaddevelopment/knee-raptor-corpus` weights, 42 temporal 2.5D windows)
  - **Engine 2:** ConvNeXt-S6 R7 (`heliosli/rsna-knee-compact-r7cnxfsss`, 4-fold S6 at 352x352 with M4 OA extent heads)
- **Fusion:** Uniform 50/50 Monotonic Percentile Rank Fusion
- **Runtime Guard:** Non-raising correlation guard (`min_rho >= 0.30`)
- **DICOM Reader:** C++ `fastread` POSIX pread threaded reader with 3D physical normal vector sorting.
- **Fail-safe / Fallbacks:** 0 study fallbacks, commit time ~450s on dual T4 GPUs.
