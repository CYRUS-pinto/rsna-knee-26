# ADR-010: Adoption of Uniform Percentile Rank Fusion and Efficiency Prize Optimization

## Status
Accepted

## Context
Post-processing experiments (V7 through V11) applying manual threshold shifts and heuristic priors overfitted severely to the 58 Gold studies and degraded public leaderboard scores. Simultaneously, the competition offers an **$18,000 Efficiency Prize** for top models executing within $T \le 500$ seconds.

## Decision
1. Permanently ban manual additive priors and asymmetric heuristic rules.
2. Enforce **Uniform 1/3 Percentile Rank Fusion** as the sole ensembling mechanism across orthogonal engines.
3. Enforce an internal inference runtime budget of **$T \le 450$ seconds** (50s safety buffer below the 500s host limit).
4. Lock **Version 6 (`57165672`, 0.956)** as Final Selection 1 on Kaggle.

## Consequences
- **Pros:**
  - Scale-invariant ensembling eliminates calibration mismatches between disparate model backbones.
  - Minimizes rank variance across 1,300 test studies.
  - Maintains direct eligibility for the $18,000 Efficiency Prize while climbing the global leaderboard.
- **Cons:**
  - Percentile rank transform requires all predictions for a test cohort to be collected in memory before ranking. On 1,300 studies $\times$ 12 floats, the matrix is $1,300 \times 12 \times 4\text{ bytes} \approx 62.4\text{ KB}$, which is negligible.
