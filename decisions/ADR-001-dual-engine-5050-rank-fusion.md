# ADR-001: Dual-Engine Uniform Percentile Rank Fusion

## Status
**Accepted** (Implemented in V6 Personal Best: `0.956`)

## Context
The competition metric is **Macro-Averaged ROC AUC** across 12 distinct abnormalities.
Our primary inference pipelines consist of two fundamentally diverse architectures:
1. **Engine 1 (Raptor CoAtNet-384 SWA)**: Global multi-instance vision transformer evaluating 42 temporal 2.5D windows.
2. **Engine 2 (Heliosli ConvNeXt-S6 R7)**: Dense pure convolutional network with 4 folds at 352x352, fluid volumetry, and dedicated OA extent heads.

In Version 7, we tested finding-specific asymmetric weighting based on anatomical domain hypotheses (e.g. assigning 65% weight to CoAtNet on ligament tears and 70% to ConvNeXt on osteoarthritis). Public LB shifted from 0.956 to 0.955 (a 0.001 shift, which is within expected test sampling noise).

## Theoretical Rationale
Let $r_1$ and $r_2$ denote the empirical percentile ranks from Engine 1 and Engine 2:
$$r_1 = \frac{\text{argsort}(p_1)}{N}, \quad r_2 = \frac{\text{argsort}(p_2)}{N}$$

For a linear combination $r(w) = w r_1 + (1 - w) r_2$, the variance of the blended score estimator is:
$$\operatorname{Var}(r(w)) = w^2 \sigma_1^2 + (1 - w)^2 \sigma_2^2 + 2w(1 - w)\operatorname{Cov}(r_1, r_2)$$

Under approximately equal model performance and rank dispersion ($\sigma_1 \approx \sigma_2 = \sigma$):
$$\operatorname{Var}(r(w)) = \sigma^2 \left[ w^2 + (1 - w)^2 + 2w(1 - w)\rho \right]$$

Setting the derivative to zero yields $w^* = 0.50$.
While ROC AUC is a pairwise ranking concordance metric rather than linear mean squared error, minimizing rank score variance provides a principled defense against overfitting to small test sets when neither model demonstrably dominates across all classes.

## Decision
1. Retain pure, uniform 50/50 percentile rank blending across all 12 classes as the default baseline.
2. Avoid ad-hoc class-by-class weight tuning on small validation samples.
3. Treat small public score shifts ($\pm 0.001$) as statistical noise rather than conclusive proof of weight sensitivity.
