# ADR-006: Selection of 2.5D Multi-Plane Slab Representation over Full 3D Voxel Ingestion

## Status
Accepted

## Context
Full 3D physical coordinate normalization ($\text{IOP} \times \text{IPP}$ spacing normalization) requires ingesting all 569.76 GB of raw DICOMs, heavy interpolation pipelines, and full 3D convolutions. On Kaggle dual T4 GPUs, running 1,300 test studies through native 3D architectures takes >1,800 seconds, violating the $T \le 500$s requirement for the $18,000 Efficiency Prize. Conversely, single 2D slices discard slice-to-slice continuity essential for detecting ligament and meniscus tears.

## Decision
1. Standardize on **2.5D Triplet Slabs**: Each slice window is composed of $(c-1, c, c+1)$ mapped to the 3 input channels of 2D foundation vision backbones.
2. Expand representation across **all 3 anatomical planes** (Sagittal, Coronal, Axial) up to $K=96$ slices at inference time.
3. Reject full 3D voxel interpolation for primary inference to preserve compliance with the $18,000 Efficiency Prize ($T \le 500$s).

## Consequences
- **Pros:** 
  - Compatible with pre-trained ImageNet-22k and medical vision foundation models.
  - End-to-end inference on 1,300 test studies completes in ~410s on Kaggle T4 GPUs.
  - Enables sub-40-minute training epochs per fold on free Kaggle / Colab GPUs.
- **Cons:**
  - Does not model arbitrary oblique scanner angle tilts between distinct scanning centers. Mitigated by discrete plane sequence embeddings in the aggregation head.
