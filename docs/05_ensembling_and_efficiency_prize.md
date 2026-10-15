# Pillar 5: Ensembling Strategy, Post-Processing Discipline & The $18,000 Efficiency Prize

## 1. Executive Summary & The Leaderboard Distribution
On the live Kaggle leaderboard (6,016 total teams):
- **Rank #1:** 0.967 (Tucker Arrants & Tony)
- **Top 10:** 0.964
- **Gold Cutoff:** 0.963
- **Top 100:** 0.958
- **The 0.956 Public Clump:** 927 teams sit between Rank #274 and #1,200 with **0.956** (Team `CYRUS PINTO` sits at #1,070).

Moving from 0.956 to **0.957 (+0.001)** immediately jumps past **800 teams** into Rank ~270. Reaching **0.963 (+0.007)** achieves a Gold Medal.

However, attempting to bridge this gap with post-processing tricks, threshold heuristics, or manual priors is mathematically guaranteed to fail.

---

## 2. Autopsy: Why Heuristic Priors Overfit and Drop Score

During V7 through V11, manual anatomical priors were implemented (e.g. `Contusion += 0.15 * ACL`, `MCL += 0.10 * Valgus`).
- **The Illusion:** On the 58 Gold studies, Gold AUC surged by **+0.016**.
- **The Reality:** On the Kaggle public leaderboard, the score dropped from **0.956 to 0.955**.

### The Mathematical Explanation
1. In the 58 Gold studies:
   - Fracture has only **7 positive cases**.
   - MCL has only **9 positive cases**.
   - Baker's Cyst has only **12 positive cases**.
2. Shifting continuous probabilities by $+0.15$ reorders 1 or 2 borderline cases in a set of 7 positives, yielding an artificial $+0.14$ jump in that single condition's Mann-Whitney U rank-sum.
3. On the hidden test set (1,300 studies with ~160 fractures and ~200 MCL tears), that same static shift corrupts the relative ranking of thousands of true negatives, destroying the global AUC.

**Rule:** Zero static additive priors. All interactions between anatomical structures must be learned end-to-end within the latent representation of the Attention-MIL head.

---

## 3. The Math of Ensembling: Uniform Percentile Rank Fusion

Why Uniform 1/3 Percentile Rank Fusion is mathematically superior to linear probability averaging:

$$\mathbf{r}_m = \text{Rank}\left(P_m\right) \in [0.0, 1.0]$$
$$\mathbf{r}_{\text{final}} = \frac{1}{M} \sum_{m=1}^M \mathbf{r}_m$$

1. **Scale Invariance:** Model A may output probabilities clustered in $[0.05, 0.45]$ while Model B outputs probabilities in $[0.10, 0.90]$. Linear averaging $\frac{P_A + P_B}{2}$ unintentionally gives Model B $\approx 70\%$ of the effective decision weight. Percentile rank transform normalizes both models to standard uniform distributions $U(0, 1)$, ensuring true equal weighting.
2. **ROC-AUC Invariance:** ROC-AUC is a pure rank-order statistic (Mann-Whitney U). Blending uniform percentile ranks minimizes rank score variance across divergent model distributions:
   $$\text{Var}(r_{\text{final}}) = \frac{1}{M^2} \sum_{m=1}^M \text{Var}(r_m) + \frac{2}{M^2} \sum_{i < j} \text{Cov}(r_i, r_j)$$
   When models are orthogonal (low covariance $\text{Cov} < 0.75$), rank variance is minimized.

---

## 4. The $18,000 Efficiency Prize: Our Secret Weapon

Competition Host Rules define the Efficiency Prize metric as:
$$\text{Efficiency Metric} = \frac{\text{Macro-AUC}}{\max(10^{-4}, \text{Benchmark} - \text{Macro-AUC})} + \frac{\text{RuntimeSeconds}}{32,400}$$
- **Constraint:** Inference on 1,300 hidden test studies must complete in **$T \le 500$ seconds**.
- **The Blind Spot of Top Competitors:** Many top 20 teams run massive 6-to-8-hour ensembles (combinations of 3D UNETRs, heavy Swin-Large models, and multi-round TTA) that will fail or be heavily penalized on runtime.
- **Our Tactical Advantage:**
  - V6 (PB 0.956) runs in **~438 seconds**.
  - V13 Tri-Hybrid Aligned runs in **~410 seconds** with pure FP16 inference and dynamic chunking.
  - By hitting 0.958–0.963 within a 410s window, our team is in prime contention for an **$18,000 cash prize** while other teams fight over medals.

---

## 5. Submission Selection Protocol
Under Kaggle rules, every team must manually select **two final submissions**:
1. **Selection 1 (The Bedrock Baseline):** **Version 6 (`57165672`, 0.956)** — Clean uniform 50/50 dual-engine blend. Zero priors, proven runtime (438s), locks in Top 2%.
2. **Selection 2 (The Frontier Model):** **Version 13 Tri-Hybrid Aligned** — CoAtNet + ConvNeXt + Orthogonal Engine 3 evaluated through 5-fold CV.
