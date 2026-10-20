# Model Registry — Every Trained Model Asset

**Updated:** 2026-10-20 15:40 UTC  
**Source of truth for:** which checkpoints exist, their quality, fold count, and Kaggle dataset slug.

> [!IMPORTANT]
> Any new agent: check this table FIRST before proposing a new model or blend.

## Production Models (Available for Inference)

| Engine | Architecture | Resolution | Folds Trained | Gold-58 Macro AUC | Per-Class Standouts | Kaggle Dataset Slug | Checkpoint Path | Status |
|--------|-------------|-----------|---------------|-------------------|--------------------|--------------------|----------------|--------|
| Raptor | CoAtNet + variants | 384+224 mixed | Unknown (public) | ~0.87 estimated | Best overall anchor | Public notebook (pre-blended) | N/A | Anchor |
| ConvNeXt R6 | ConvNeXt-Small | 384 | Unknown (public) | ~0.85 estimated | OA/Effusion | Public notebook | N/A | Anchor |
| Super-CoAtNet | CoAtNet | 384 | Unknown (public) | ~0.9139 Gold-58 | N/A | Public notebook | N/A | Anchor |
| E11 | ConvNeXt-Base | 384 | 1 | 0.8712 | N/A | N/A | N/A | Available |
| E12 | ConvNeXt-Large-384 | 384 | **1 (SINGLE FOLD)** | **0.8808** | ACL: 0.92+, MedOA: 0.96+ | `cyruspinto/rsna-knee-engine12-convnext-large-weights` | 333 MB | Ready |
| E13 | Swin-Base-384 | 384 | **1 (SINGLE FOLD)** | **0.8577** | ACL: 0.9142, MedOA: 0.9581, Fracture: 0.9042 | `cyruspinto/rsna-knee-engine13-swin-base-weights` | 332.6 MB | Ready |

## Models NOT Yet Trained (Priority List)

| Priority | Architecture | Resolution | Folds Needed | Estimated Training Time | Estimated Impact |
|----------|-------------|-----------|-------------|------------------------|-----------------|
| 1 | ConvNeXt-Large-384 | 384 | 5 (only 1 done) | 12-15h on A100 (4 more folds) | +0.003-0.005 AUC |
| 2 | Swin-Base-384 | 384 | 5 (only 1 done) | 16-20h on T4 (4 more folds) | +0.002-0.004 AUC |
| 3 | DINOv2-ViT-L/14 | 384 | Feature extraction + head | 4-6h total | +0.001-0.003 AUC |
| 4 | EfficientNet-V2-M | 384 | 5 | 10-15h on A100 | +0.001-0.002 AUC |
| 5 | MaxViT-Base | 384 | 5 | 15-20h on A100 | +0.001-0.002 AUC |

## Model Diversity Matrix (Pairwise Rank Correlation)

> [!WARNING]
> We have NOT computed OOF rank correlations yet because we lack multi-fold OOF predictions. This is a critical gap.

| | Raptor | ConvNeXt R6 | E12 | E13 |
|--|--------|------------|-----|-----|
| Raptor | 1.00 | ~0.95+ (est.) | Unknown | Unknown |
| ConvNeXt R6 | ~0.95+ | 1.00 | Unknown | Unknown |
| E12 | Unknown | Unknown | 1.00 | Unknown |
| E13 | Unknown | Unknown | Unknown | 1.00 |

**Problem:** Raptor and ConvNeXt R6 are likely ρ > 0.95 (same family, same training data). This means blending them adds almost nothing. We need TRULY orthogonal models.

## Blend History (Weight Allocations)

| Candidate | Raptor 384 | Raptor 224 | ConvNeXt R6 | E11 | E12 | E13 | Score |
|-----------|-----------|-----------|------------|-----|-----|-----|-------|
| V17 (PB) | ~50% | ~20% | ~30% | — | — | — | 0.958 |
| V45 | ~70% | included | ~30% | — | — | — | 0.958 |
| V47 | ~40% | ~18% (double!) | ~17% | — | ~25% | — | 0.953 |
| V48 | ~70% | — | — | ~15% | ~15% | ~15% | PENDING |
