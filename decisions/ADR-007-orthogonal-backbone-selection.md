# ADR-007: Selection of Orthogonal Model Architecture (DINOv2 / Swin) for Engine 3

## Status
Accepted

## Context
Our existing dual-engine ensemble consists of:
1. Raptor CoAtNet-3 (Hybrid CNN-Transformer with relative attention).
2. Heliosli ConvNeXt-Base R7 (Pure modern ConvNet).

Adding another model from the same families (e.g. CoAtNet-4 or ConvNeXt-Large) results in high residual error correlation ($\rho > 0.90$), yielding diminishing returns on public/private leaderboard metrics. We require a model with an orthogonal inductive bias.

## Decision
1. Select **DINOv2 (ViT-Base)** as the primary orthogonal architecture candidate for Engine 3, and **Swin Transformer** as the secondary candidate.
2. Train models strictly using the 5-fold Stratified Group-KFold split (`data/train_folds.parquet`).
3. Reject additional variants of CoAtNet or standard ResNets for Engine 3.

## Consequences
- **Pros:**
  - DINOv2 provides self-supervised global patch attention, capturing non-local relationships across the knee joint that convolutions miss.
  - Significantly lowers ensemble error covariance ($\rho < 0.75$), maximizing the theoretical gain of uniform rank blending.
  - Model weights and inference overhead fit well within the 16 GB VRAM and $T \le 450$s execution envelope.
- **Cons:**
  - ViTs require careful learning rate warmup (e.g., peak LR $\le 10^{-4}$) to prevent divergence.
