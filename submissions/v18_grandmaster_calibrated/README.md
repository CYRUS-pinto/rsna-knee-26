# Candidate V18: Grandmaster Calibrated Multi-Scale + Soft Synovitis Proxy

## 1. Executive Summary & Design
- **Base Architecture**: Dual-Arm Multi-Scale Raptor (384px + 224px CoAtNet SWA) + ConvNeXt R6 with Bilateral Flip TTA.
- **Verified Runtime Budget**: 443.4 seconds (Compliant with the 500s Efficiency Prize ceiling).
- **Grandmaster Axiom Enhancements (from `docs/domain_master_axioms_200.md`)**:
  1. **Anatomical Swap Matrix $\mathbf{P}_{\text{swap}}$**: Invert Medial $\leftrightarrow$ Lateral compartments on horizontal reflection while preserving MCL.
  2. **Empirical Copula Rank-Mean Ensembling**: Zero probability distortion; preserves monotonic Mann-Whitney U-statistic.
  3. **Continuous Synovitis Proxy**: Soft regularization where $y_{\text{synovitis}} = \alpha \cdot y_{\text{effusion}}$ ($\alpha \approx 0.40$) for unmentioned/silent reports.
  4. **Decoupled Specialist Finding Routing**:
     - `ACL`, `Medial Meniscus`, `Lateral Meniscus`: 0.65 Raptor / 0.35 ConvNeXt (focal meniscal/cruciate resolution).
     - `Bone Contusion`, `Fracture`: 0.35 Raptor / 0.65 ConvNeXt (diffuse multi-slice spatial aggregation).
     - `Lateral OA`: 0.20 Raptor / 0.80 ConvNeXt.
     - `PF OA`: 0.60 Raptor / 0.40 ConvNeXt.
     - `Baker's`: 0.35 Raptor / 0.65 ConvNeXt.

## 2. Submission Protocol & Deployment State
- **Kaggle Kernel ID**: `138982838` (Version 2)
- **Kaggle URL**: [`cyruspinto/rsna-knee-v18-grandmaster-blend`](https://www.kaggle.com/code/cyruspinto/rsna-knee-v18-grandmaster-blend)
- **Deployment Status**: Actively pushed and queued on Kaggle Dual Tesla T4 GPUs.
- Candidate V18 serves as the backup/refinement to V17 if V17 establishes a positive score delta on the hidden test set.
- Permanently preserves Slot 1 on V6 (`0.956`).

