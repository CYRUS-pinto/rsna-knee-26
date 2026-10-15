# ADR-008: Adoption of Multi-Plane Attention-MIL with Discrete Sequence Plane Embeddings

## Status
Accepted

## Context
Knee MRI studies contain localized findings visible only in specific planes (e.g. MCL in Coronal, ACL in Sagittal, Patellofemoral in Axial). Standard global pooling either washes out focal pathology (GAP) or introduces noise vulnerability (GMP). Furthermore, naively concatenating 96 slices into a 1D sequence corrupts positional embeddings across acquisition boundaries.

## Decision
1. Implement **12-Class Condition-Specific Attention-MIL Pooling**: Compute independent attention weight vectors for each of the 12 target findings.
2. Incorporate **Discrete Sequence Plane Embeddings**: Slices are annotated with their anatomical sequence plane ID ($0 \dots 4$), combined with relative within-sequence depth.
3. Apply a 2-layer Transformer Encoder across slice representations prior to class-specific attention pooling.

## Consequences
- **Pros:**
  - Enables models trained on variable slice counts ($K=44$ in training, $K=96$ at inference) to adapt dynamically without dimension mismatch.
  - Allows each pathological head to attend exclusively to the clinically relevant sequence plane.
  - Mathematically prevents gradient dilution on rare, localized tears.
- **Cons:**
  - Adds ~1.2M parameters to the model head (negligible runtime impact: < 2ms per study).
