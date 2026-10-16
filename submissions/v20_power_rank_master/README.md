# Candidate V20: Generalized Power-Rank Multi-Scale Master Blend

## 1. Executive Summary & Design
- **Base Architecture**: Identical to verified **Candidate V17** (which scored **0.958 PB** in 443.4s offline).
  - Multi-Scale Raptor Arm: 65% CoAtNet-384 + 35% CoAtNet-224 SWA.
  - ConvNeXt R6 Arm: 4-fold fluid stackers with Bilateral Horizontal Flip TTA.
- **Empirical Re-Weighting (Preserved from V17)**:
  - Lateral OA: 0.20 Raptor / 0.80 ConvNeXt
  - PF OA: 0.60 Raptor / 0.40 ConvNeXt
  - Synovitis: 0.80 Raptor / 0.20 ConvNeXt
  - Baker's: 0.35 Raptor / 0.65 ConvNeXt
  - Medial OA: 0.50 Raptor / 0.50 ConvNeXt
- **Core Engineering Innovation (Generalized Power-Mean Ensembling)**:
  - For acute focal pathologies (`ACL`, `Medial Meniscus`, `Lateral Meniscus`, `Fracture`), replaces arithmetic rank blending with **Power-Mean ($p = 1.4$)**:
    $$R_{\text{blend}}(c) = \left( w_r(c) \cdot R_{\text{Raptor}}(c)^p + (1 - w_r(c)) \cdot R_{\text{ConvNeXt}}(c)^p \right)^{1/p}$$
  - Smoothly amplifies concordant high-confidence signals where tears are subtle and localized.
  - **Zero hand-crafted if/then floors. Zero threshold clipping.** 100% mathematically continuous.
- **Runtime & Efficiency**:
  - Compute overhead of power-mean is $< 0.01$s. Total runtime remains identical to V17 (~443s), defending the **$18,000 Efficiency Prize** ceiling.
