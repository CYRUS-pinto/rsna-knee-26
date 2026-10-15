# ADR-004: Train vs Inference Slice Distribution Alignment (44 Slices vs 96 Slices)

## Status
**Accepted**

## Context
A major risk identified in the training pipeline is the representation discrepancy between the precomputed training corpus and the production inference pipeline:
1. **Training Corpus (`dreaddevelopment/knee-raptor-corpus`)**:
   - Contains pre-extracted volumes with **44 slices** per study at **336x336** resolution.
   - Slices are predominantly sampled from the sagittal fluid-sensitive series.
2. **Inference Pipeline (`scratch/v12_clean_offline/` and V6 PB)**:
   - Constructs a standardized **5-slot anatomical tensor with 96 slices** across all three imaging planes:
     - Slot 1: Sagittal Fluid (26 slices)
     - Slot 2: Sagittal Non-Fluid (22 slices)
     - Slot 3: Coronal Fluid (18 slices)
     - Slot 4: Coronal Non-Fluid (12 slices)
     - Slot 5: Axial (18 slices)
   - Resampled at native $384 \times 384$ or $352 \times 352$.

## Risk Analysis: Positional Domain Shift
When training a 3D sequential transformer head with 1D slice positional embeddings along the Z-axis:
- Slices $0 \dots 15$ during training correspond strictly to sagittal joint depths.
- During inference, slices $0 \dots 95$ span sagittal, coronal, and axial sequences sequentially.
- If a transformer head learns that "slice 10" is mid-sagittal ACL, feeding it coronal MCL slices at position 30 will cause severe positional misattribution!

## Decision
1. **Model Architecture Selection**:
   - For models trained on the 44-slice corpus, use **Permutation-Invariant Attention-MIL** (`--head mil`). Because Attention-MIL pools features using finding-specific queries without assuming fixed slice-index semantics, it generalizes safely across variable slice depths and planes.
2. **Transformer Head Deployment Constraint**:
   - 3D Slice-Transformers (`--head transformer`) must only be deployed on models where the training dataset explicitly mirrors the exact 5-slot 96-slice multi-plane tensor used during test-time inference.
3. **Primary Recommendation**:
   - Execute the stabilized Attention-MIL + BCE baseline first to establish verified convergence before experimenting with sequential multi-slice transformers.
