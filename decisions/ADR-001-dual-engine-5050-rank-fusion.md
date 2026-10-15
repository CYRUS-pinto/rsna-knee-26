# ADR-001: Dual-Engine Uniform Percentile Rank Fusion

## Status
**Accepted** (Implemented in V6 Personal Best: `0.956`)

## Context
The competition metric is **Macro-Averaged ROC AUC** across 12 distinct abnormalities.
Our primary inference pipelines consist of two fundamentally diverse architectures:
1. **Engine 1 (Raptor CoAtNet-384 SWA)**: Global multi-instance vision transformer evaluating 42 temporal 2.5D windows.
2. **Engine 2 (Heliosli ConvNeXt-S6 R7)**: Dense pure convolutional network with 4 folds at 352x352, fluid volumetry, and dedicated OA extent heads.

In Version 7, we attempted finding-specific asymmetric weighting based on anatomical domain knowledge (e.g. assigning 65% weight to CoAtNet on ligament tears and 70% to ConvNeXt on osteoarthritis). This resulted in an immediate leaderboard regression from **0.956 to 0.955**.

## Mathematical Proof of Decision
Let $r_1$ and $r_2$ denote the empirical percentile ranks from Engine 1 and Engine 2 respectively:
$$r_1 = \frac{\text{argsort}(p_1)}{N}, \quad r_2 = \frac{\text{argsort}(p_2)}{N}$$

For any convex linear combination $r(w) = w r_1 + (1 - w) r_2$, the variance of the ensemble rank estimator is:
$$\operatorname{Var}(r(w)) = w^2 \sigma_1^2 + (1 - w)^2 \sigma_2^2 + 2w(1 - w)\operatorname{Cov}(r_1, r_2)$$

Since percentile ranks on uniform distributions have equal variance $\sigma_1^2 = \sigma_2^2 = \sigma^2$:
$$\operatorname{Var}(r(w)) = \sigma^2 \left[ w^2 + (1 - w)^2 + 2w(1 - w)\rho \right]$$

Differentiating with respect to $w$ and setting to zero:
$$\frac{d}{dw}\operatorname{Var}(r(w)) = \sigma^2 \left[ 2w - 2(1 - w) + 2(1 - 2w)\rho \right] = 2\sigma^2 (1 - \rho)(2w - 1) = 0$$

Since the correlation between orthogonal models satisfies $\rho < 1$, the unique global minimum variance occurs strictly at:
$$w^* = 0.50$$

## Decision
1. Retain pure, uniform 50/50 percentile rank blending across all 12 classes as the immutable baseline.
2. Reject class-specific weight heuristic sweeps on the two-model ensemble.
3. Performance breakthroughs beyond 0.956 require adding a genuine **Engine 3** model family rather than perturbing weights.
