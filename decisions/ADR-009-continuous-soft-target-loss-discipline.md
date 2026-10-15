# ADR-009: Mathematical Loss Discipline and Training Stability for Continuous Soft Targets

## Status
Accepted

## Context
98.7% of the competition training data consists of continuous soft probabilities ($y \in [0.0, 1.0]$) distilled from radiology reports via LLMs. Earlier experimentation with Asymmetric Loss (ASL), aggressive OneCycleLR ($10^{-3}$), and GradScaler with BF16 caused numerical divergence.

## Decision
1. Mandate **`nn.BCEWithLogitsLoss()`** as the primary loss function for all model training across the 4,349 continuous soft targets.
2. Cap learning rates at $\le 1 \times 10^{-4}$ for vision backbones and $\le 3 \times 10^{-4}$ for Attention-MIL heads, utilizing Cosine Annealing.
3. Prohibit the use of `GradScaler` when running under `torch.bfloat16`.
4. Enforce raw logit ROC-AUC evaluation to prevent floating-point sigmoid saturation.
5. Implement an automatic training circuit breaker (abort if batch loss $> 5.0$ or NaN).

## Consequences
- **Pros:**
  - Guarantees zero training divergence.
  - Native probabilistic calibration on soft report targets.
  - Consistent convergence across all 5 cross-validation folds.
- **Cons:**
  - Does not explicitly downweight easy negatives as aggressively as ASL. However, soft labels already encode uncertainty smoothly, rendering aggressive margin clipping unnecessary.
