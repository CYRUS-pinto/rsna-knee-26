# Submission Versions 9, 10, 11: Heuristic Clinical Priors & Parameter Sweeps

## Summary & Scores
- **Public LB Scores:**
  - **Version 9:** `0.955` (-0.001 from V6 PB)
  - **Version 10:** `0.955`
  - **Version 11:** `0.955`
- **Configuration Tested:** Clinical conditional rules applied post-rank blending in Cell 6:
  - `Contusion = Contusion + alpha * ACL` ($\alpha \in [0.10, 0.20]$)
  - `Synovitis = Synovitis + beta * Effusion` ($\beta \in [0.15, 0.25]$)
  - `Baker's Cyst = Baker's Cyst + gamma * Effusion`
  - `Medial Meniscus = Medial Meniscus + delta * Medial OA`

## Why It Failed (Empirical & Mathematical Takeaway)
1. **Local vs Public Divergence:** The rules raised Gold-58 Macro-AUC from 0.917 to 0.933 (+0.016 gain locally), but degraded Public LB by -0.001 across all 3 submissions.
2. **Discordant Inversion:** Patients with acute contusion without ACL tears were penalized and ranked below patients with chronic ACL tears without contusion.
3. **Sample Size Trap:** With only 7-12 positive cases for rare findings in the 58 Gold studies, parameter tuning overfit binomial sampling noise.
4. **Resolution:** All heuristic priors permanently excised in V12.
