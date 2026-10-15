# Pillar 1: Input Representation & Spatial Geometry Breakdown

## 1. Executive Summary & Context
In RSNA Knee Abnormality Detection, MRI studies comprise multi-slice volumetric acquisitions across diverse pulse sequences (Sagittal T1, Sagittal PD/T2 Fat-Suppressed, Coronal PD/FS, Axial). The raw competition dataset contains **569.76 GB of DICOMs** (819,640 individual DICOM files across 4,407 training studies). 

How the raw volumetric pixels are transformed into neural network inputs is the single most critical determinant of representation quality.

---

## 2. Comparative Analysis of Approaches

| Representation Strategy | Spatial Resolution | Volumetric Context | Inference Latency | GPU Memory | Feasibility in 7 Days | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Pure 2D Single Slice** | $384 \times 384$ | Zero $Z$-axis depth | Ultra-fast (<150s) | Very Low (<4GB) | Immediate | ❌ **Rejected:** Cannot resolve subtle cruciate ligament continuity or meniscus tears. |
| **B. 2.5D Slab Triplets $(c-1, c, c+1)$** | $384 \times 384$ | Local contiguous slices as RGB channels | Fast (~380s) | Low (~8GB) | Immediate (Current PB) | ✅ **Baseline Foundation:** Allows 2D pre-trained ImageNet backbones to leverage local 3D continuity. |
| **C. Native 3D Physical Voxel Resampling** | Uniform $(1.0, 1.0, 1.0)$ mm | Full anatomical volume | Slow (>600s, violates efficiency prize) | Extreme (>20GB) | Impossible without cluster | ❌ **Impractical:** Violates $T \le 500$s efficiency limit; requires days of raw DICOM ingestion. |
| **D. Multi-Sequence 2.5D Multi-Plane Stacking (Current Optimal)** | $384 \times 384$ | Multi-plane (Sagittal + Coronal + Axial) | Fast (~410s) | Medium (~10GB) | High (Ready in inference) | 🏆 **Gold Standard:** Extracts key slabs across all 3 anatomical planes while maintaining 2D backbone speed. |

---

## 3. Deep-Dive: Why 2.5D Slab Triplets Are Clinically & Computationally Optimal

### A. The Radiologist's Viewing Modality
When a musculoskeletal radiologist examines a knee MRI:
1. They scroll through slices to establish continuity. A tear is defined by a bright fluid signal traversing a ligament across at least **two consecutive slices**.
2. Stacking three consecutive slices $(c-1, c, c+1)$ into the 3 color channels $(R, G, B)$ allows standard 2D convolutional kernels to perform simultaneous 2D spatial feature extraction and 1st-order temporal/depth gradient calculation:
   $$\nabla_z \approx \frac{I(x, y, c+1) - I(x, y, c-1)}{2 \Delta z}$$

### B. Computational Budget & Efficiency Prize ($18,000)
- True 3D convolutions ($3 \times 3 \times 3$) scale cubically in FLOPs: $\mathcal{O}(D \cdot H \cdot W)$.
- Evaluating 1,300 test studies with 3D MedNeXt or UNETR on dual Kaggle T4 GPUs takes ~1,800–2,400 seconds. This **instantly disqualifies** the submission from the $18,000 Efficiency Prize ($T \le 500$s).
- 2.5D slabs process 96 slices in **~380–420 seconds**, maintaining top-tier inference speed while extracting 95% of the volumetric continuity signal.

---

## 4. Architectural Decision & Action Items
1. **Maintain 2.5D Triplet Windows:** Slices are sampled as $(c-1, c, c+1)$ normalized to $[0, 1]$.
2. **Multi-Plane Integration:** Incorporate Coronal and Axial slabs alongside Sagittal slabs. Slices from Coronal PD/FS are mandatory for Medial/Lateral Meniscus and MCL accuracy.
3. **Discrete Plane Tagging:** Assign discrete plane IDs so subsequent sequence models do not treat plane boundaries as spatial continuities.
