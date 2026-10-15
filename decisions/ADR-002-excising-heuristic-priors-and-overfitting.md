# ADR-002: Excising Clinical Heuristic Priors & Preventing Small-Sample Overfitting

## Status
**Accepted** (Implemented in V12 Clean Offline Pipeline)

## Context
In Versions 9, 10, and 11, post-processing rules were introduced into Cell 6 to exploit clinical correlations:
1. `Contusion = Contusion + 0.15 * ACL`
2. `Synovitis = Synovitis + 0.20 * Effusion`
3. `Baker's Cyst = Baker's Cyst + 0.15 * Effusion`
4. `Medial Meniscus = Medial Meniscus + 0.10 * Medial OA`

Despite improving scores on the local 58 Gold studies (0.917 -> 0.933), public submissions scored 0.955 (-0.001 from V6 PB).

## Analysis & Evaluation

### 1. Public Score Impact & Statistical Noise
On an estimated ~400 public test cases, a score shift of 0.001 (0.956 vs 0.955) is within standard evaluation noise. However, because V9, V10, and V11 showed zero upward movement on public LB despite strong local gains, the local improvements did not generalize.

### 2. Risk of Ranking Inversions on Discordant Pairs
Additive heuristic rules like $p'_{\text{contusion}} = p_{\text{contusion}} + 0.15 \times p_{\text{ACL}}$ inherently assume that co-occurring conditions should always boost ranking. For discordant patients (e.g. true contusion without ACL injury vs true ACL tear with zero bone contusion), adding an uncorrelated signal risks boosting negative cases above positive cases, degrading pairwise rank concordance.

### 3. Small-Sample Contamination (The 58 Gold Studies Trap)
Breakdown of positive cases among the 58 Gold studies:
- Fracture: **7 positive cases**
- MCL: **9 positive cases**
- Baker's Cyst: **12 positive cases**

With so few positive examples for rare findings, small-sample variance is high: a single threshold shift in a 7-positive class moves observed AUC by $\approx 0.0028$. Tuning continuous post-processing multipliers to optimize the 58 Gold studies risks fitting small-sample variance.

## Decision
1. Permanently excise all 4 heuristic conditional rules from the submission pipeline.
2. Treat the 58 Gold studies strictly as a **coarse fail-safe / bug detector** (verifying that decoders did not invert axes or crash), never as a fine hyperparameter tuner.
3. Retain pure percentile rank blending without post-hoc heuristic additions.
