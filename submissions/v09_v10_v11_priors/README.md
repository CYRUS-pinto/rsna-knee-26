# Submission Versions 9, 10, 11: Heuristic Clinical Priors & Parameter Sweeps

## Summary & Scores
- **Public LB Scores:**
  - **Version 9:** `0.955` (-0.001 from V6 PB 0.956)
  - **Version 10:** `0.955`
  - **Version 11:** `0.955`
- **Configuration Tested:** Post-processing heuristics applied after rank blending in Cell 6:
  - `Contusion = Contusion + alpha * ACL`
  - `Synovitis = Synovitis + beta * Effusion`
  - `Baker's Cyst = Baker's Cyst + gamma * Effusion`
  - `Medial Meniscus = Medial Meniscus + delta * Medial OA`

## Empirical Findings & Evaluation
1. **Public LB Impact:** Across three parameter variations (V9, V10, V11), Public LB remained at 0.955. On an estimated ~400 public test cases, this 0.001 difference from V6 (0.956) is within statistical noise.
2. **Local vs Public Divergence:** The priors improved the score on the 58 Gold studies (0.917 to 0.933), but did not translate to higher public leaderboard scores. This supports the conclusion that the 58 Gold studies are too small (only 7-12 positives for rare findings) to use as an optimization set.
3. **Decision:** Because the heuristics added complexity without any verified public score improvement, they were excised in V12 in favor of the simpler, unadjusted ensemble.
