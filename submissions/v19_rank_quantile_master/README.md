# Candidate V19: Rank-Quantile Grandmaster Master Blend

## 1. Executive Summary & Design
- **Base Architecture**: Dual-Arm Multi-Scale Raptor (384px + 224px CoAtNet SWA) + ConvNeXt R6 with Bilateral Flip TTA.
- **Verified Runtime Budget**: ~460 seconds (Strictly compliant with $18k Efficiency Prize ceiling $T \le 500$s).
- **Core Engineering Upgrades (from `ADR-018` & `docs/domain_master_axioms_200.md`)**:
  1. **Full-Spectrum Rank-Quantile Uniformization**:
     - Maps each blended finding column $j$ to a uniform empirical quantile distribution: $\tilde{p}_j = \text{pct}(b_j)$.
     - Eliminates logit scale compression around 0.5.
     - **Immunizes against test-set prevalence shifts** highlighted in the official RSNA dataset description.
  2. **Asymmetric Pathology Routing**:
     - `ACL`, `Medial Meniscus`, `Lateral Meniscus`: 0.65 Raptor / 0.35 ConvNeXt (resolving focal micro-tears).
     - `Bone Contusion`, `Fracture`: 0.35 Raptor / 0.65 ConvNeXt (diffuse multi-slice spatial aggregation).
     - `Lateral OA`: 0.20 Raptor / 0.80 ConvNeXt.
     - `PF OA`: 0.60 Raptor / 0.40 ConvNeXt.
     - `Synovitis`: 0.80 Raptor / 0.20 ConvNeXt.
     - `Baker's`: 0.35 Raptor / 0.65 ConvNeXt.
  3. **Clinical Pathological Co-Occurrence Network**:
     - **Synovitis / Effusion Floor**: $r_{\text{synovitis}} \ge 0.38 \cdot r_{\text{effusion}}$ (Axioms 32–34 & 65).
     - **PF OA Bi-Compartmental Progression Floor**: $r_{\text{PF OA}} \ge 0.25 \cdot \sqrt{r_{\text{Medial OA}} \cdot r_{\text{Lateral OA}}}$ (mechanical axis deviation load transfer).
     - **Acute Trauma Meniscus Floor**: Where $\text{ACL} > 0.85$ and $\text{Contusion} > 0.80$, ensure $\max(\text{MM}, \text{LM}) \ge 0.35$.
  4. **Anatomical Swap Matrix $\mathbf{P}_{\text{swap}}$**: Inverts Medial $\leftrightarrow$ Lateral compartments on horizontal reflection while preserving MCL.

## 2. Submission & Kernel Details
- **Kaggle Kernel ID**: `cyruspinto/rsna-knee-v19-rank-quantile-blend`
- **Machine Shape**: NvidiaTeslaT4 (Dual T4)
- **Role in Portfolio**: Primary high-upside alternative for **Slot 2** (Slot 1 permanently locked to V6 `0.956`).
