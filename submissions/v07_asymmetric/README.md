# Submission Version 7: Asymmetric Weights Experiment (Public LB: 0.955)

## Overview
- **Kaggle Ref:** `57172886`
- **Public LB Score:** **0.955** (-0.001 regression from V6)
- **Hypothesis:** Give CoAtNet 65% weight on acute tears (ACL, MCL, Meniscus) and ConvNeXt 70% weight on Osteoarthritis (Medial/Lateral/PF OA) based on anatomical strengths.
- **Empirical Result:** Public LB dropped to 0.955.
- **Takeaway:** Uniform 50/50 rank blend is mathematically minimum-variance for two models ($Var = \frac{1}{4}(\sigma_1^2 + \sigma_2^2 + 2\rho\sigma_1\sigma_2)$). Skewing weights increases variance unless the individual model is substantially superior. Advancing to 0.960+ requires adding an orthogonal Engine 3, not re-weighting Engines 1 & 2.
