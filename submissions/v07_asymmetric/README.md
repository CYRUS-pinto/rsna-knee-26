# Submission Version 7: Asymmetric Weights Experiment (Public LB: 0.955)

## Overview
- **Kaggle Ref:** `57172886`
- **Public LB Score:** **0.955** (-0.001 from V6 PB 0.956)
- **Configuration Tested:** Finding-specific blending weights (CoAtNet 65% on acute tears, ConvNeXt 70% on OA) based on anatomical strengths.
- **Empirical Result:** Public LB score was 0.955. On an estimated ~400 public test cases, a 0.001 shift is within expected evaluation noise.
- **Decision:** Because asymmetric weighting showed no empirical gain on the leaderboard, the simpler uniform 50/50 rank blend was retained as the default.
