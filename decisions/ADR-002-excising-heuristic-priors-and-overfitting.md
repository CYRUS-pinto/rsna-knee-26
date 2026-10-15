# ADR-002: Excising Clinical Heuristic Priors & Preventing Small-Sample Overfitting

## Status
**Accepted** (Implemented in V12 Clean Offline Pipeline)

## Context
In Versions 9, 10, and 11, post-processing rules were introduced into Cell 6 to exploit clinical correlations:
1. `Contusion = Contusion + 0.15 * ACL`
2. `Synovitis = Synovitis + 0.20 * Effusion`
3. `Baker's Cyst = Baker's Cyst + 0.15 * Effusion`
4. `Medial Meniscus = Medial Meniscus + 0.10 * Medial OA`

Despite improving scores on the local 58 Gold studies, live submissions plateaued at **0.955** (-0.001 regression from V6 PB).

## Forensic Analysis

### 1. Inversion of Discordant Pairs
Consider two patients $A$ and $B$:
- Patient $A$: Severe acute Bone Contusion with an intact ACL ($y_{\text{contusion}} = 1, y_{\text{ACL}} = 0$). Engine predictions: $p_{\text{contusion}} = 0.65, p_{\text{ACL}} = 0.10$.
- Patient $B$: Chronic ACL tear with zero bone marrow edema ($y_{\text{contusion}} = 0, y_{\text{ACL}} = 1$). Engine predictions: $p_{\text{contusion}} = 0.55, p_{\text{ACL}} = 0.95$.

Applying the heuristic $p'_{\text{contusion}} = p_{\text{contusion}} + 0.15 \times p_{\text{ACL}}$:
- Patient $A$: $p'_{\text{contusion}} = 0.65 + 0.15(0.10) = 0.665$
- Patient $B$: $p'_{\text{contusion}} = 0.55 + 0.15(0.95) = 0.6925$

**Outcome:** Patient $B$ (true negative for contusion) is boosted higher than Patient $A$ (true positive for contusion). Because ROC AUC is strictly the Wilcoxon-Mann-Whitney probability of correct ranking:
$$\text{AUC} = \frac{1}{N^+ N^-} \sum_{i \in \text{Pos}} \sum_{j \in \text{Neg}} \mathbb{I}(p_i > p_j)$$
Every discordant inversion subtracts $\frac{1}{N^+ N^-}$ directly from the target AUC.

### 2. Small-Sample Contamination (The 58 Gold Studies Trap)
The 58 Gold studies were used both to tune soft consensus labels and to validate post-processing heuristics.
Breakdown of positive cases among the 58 Gold studies:
- Fracture: **7 positive cases**
- MCL: **9 positive cases**
- Baker's Cyst: **12 positive cases**

A single false positive flip in a class with 7 positives changes the observed validation AUC by $\Delta \text{AUC} \approx \frac{1}{7 \times 51} \approx 0.0028$. Tuning hyperparameters to optimize the 58 Gold studies is mathematically guaranteed to fit high-frequency binomial noise.

## Decision
1. Permanently excise all 4 heuristic conditional rules from the submission pipeline.
2. Treat the 58 Gold studies strictly as a **coarse fail-safe / bug detector** (verifying that no model has inverted axes or crashed outputs), never as a fine-grained hyperparameter optimizer.
3. Lock live Kaggle submissions until a candidate model demonstrates statistically significant offline validation on the broader 4,349 cross-validated studies.
