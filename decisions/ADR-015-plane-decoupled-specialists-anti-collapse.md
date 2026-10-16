# ADR-015: Plane-Decoupled Specialist Architecture to Eliminate Multi-Task Representation Collapse

## Status
Accepted (Phase 2 Architectural Standard)

## Context & Problem Statement
During the empirical audit of Engine 4 (ConvNeXt-Small Multi-Plane MIL), an evaluation of the out-of-fold prediction covariance revealed complete representation collapse:
- **Mean off-diagonal correlation among predicted columns:** **`0.9966`** (compared to `0.1795` among ground-truth labels).
- **Cross-AUC Inversion:** The `Contusion` prediction head achieved a higher AUC on `Synovitis` ($0.824$) than the `Synovitis` head itself ($0.823$).
- **Root Cause:** In multi-task learning across 12 pathologies, diffuse high-prevalence findings (Effusion at ~60%, Osteoarthritis at ~40%) generate loss gradients $10\times$ larger than focal acute tears (ACL at 16%, MCL at 3%). Because all 12 heads shared a single feature extractor optimized at a uniform learning rate ($1e-4$), the backbone aligned entirely with the dominant principal component of "Total Joint Disease Burden."

## Decision
We formally decouple model training into **Three Plane-Decoupled Specialist Architectures**:

### 1. Model A: Sagittal Acute Tear Specialist
- **Sequence Inputs:** Sagittal Fluid-Sensitive (Slot 0) + Sagittal Non-Fluid (Slot 1).
- **Targets:** `ACL`, `Medial Meniscus`, `Lateral Meniscus`, `Bone Contusion`.
- **Loss:** Asymmetric Focal BCE ($\gamma=2, \alpha=0.75$) with positive weighting on acute tears.
- **Backbone LR:** $1e-5$ with 1-epoch frozen warmup; Head LR: $1e-3$.

### 2. Model B: Coronal Degenerative Specialist
- **Sequence Inputs:** Coronal Fluid-Sensitive (Slot 2) + Coronal Non-Fluid (Slot 3).
- **Targets:** `MCL`, `Medial OA`, `Lateral OA`.
- **Loss:** Multi-label BCE + Joint Space Ordinal Margin Loss.

### 3. Model C: Axial & Volumetric Fluid Specialist
- **Sequence Inputs:** Axial (Slot 4) + Fluid-Sensitive Fat-Sat volumes.
- **Targets:** `PF OA`, `Joint Effusion`, `Synovitis`, `Baker's Cyst`, `Fracture`.
- **Loss:** Soft BCE calibrated on JEV continuous report labels.

## Consequences
- **Positive:** Mathematically guarantees zero gradient interference between diffuse effusion and sparse focal tears. Eliminates representation collapse ($r \le 0.40$ expected across groups).
- **Positive:** Each specialist model operates at its native optimal slice resolution and window depth.
- **Latency Budget:** Inference across 3 decoupled light models on Dual T4 GPUs remains well under the 450s limit.
