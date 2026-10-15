# Submission Candidate V15: Multi-Scale Raptor + ConvNeXt R7 with Anatomical TTA

## 1. Overview
- **Kaggle Kernel:** `cyruspinto/rsna-knee-multiscale-tta`
- **Target:** Evaluating impact of native Anatomical Test-Time Augmentation on 4-fold ConvNeXt R7 alongside Multi-Scale CoAtNet SWA.
- **Components:**
  - **Stage 1A:** CoAtNet-384 SWA ($65\%$)
  - **Stage 1B:** CoAtNet-224 SWA ($35\%$)
  - **Stage 2:** ConvNeXt-S6 R7 with `FLIP_TTA = True` (dual-pass with anatomical permutation mapping).
  - **Stage 3:** Non-parametric percentile rank fusion:
    $$r_{\text{raptor}} = 0.65 \cdot \text{Rank}(S_{384}) + 0.35 \cdot \text{Rank}(S_{224})$$
    $$R_{\text{final}} = 0.50 \cdot r_{\text{raptor}} + 0.50 \cdot \text{Rank}(S_{\text{ours\_tta}})$$

## 2. Hypothesis & Gate
- **Hypothesis:** Bilateral test-time augmentation on R7 stabilizes edge activations on subtle tears (ACL/MCL/meniscus) without degrading runtime budget.
- **Latency Budget:** TTA increases ConvNeXt execution from ~359s to ~450s, bringing total wall-clock to **~560s** (well within Kaggle's 9h limit).
- **Gate Protocol:** If V15 Public LB score beats V14, lock V15 as Slot 2 candidate.
