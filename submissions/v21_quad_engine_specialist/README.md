# Candidate V21: Quad-Engine Multi-Plane Specialist Blend

## Architecture Summary
Candidate V21 extends the verified Candidate V17 (`0.958` Public LB) and Candidate V20 architectures by adding a 4th specialized neural engine: **Engine 5 Plane-Decoupled Specialists**.

### Engine Hierarchy
1. **Engine 1 (Raptor-384)**: CoAtNet-d4 whole-volume model trained on 384x384 resolution with SWA weights (`nartaa/rsna-knee-publication-swa-weights-20261007`).
2. **Engine 2 (Raptor-224)**: CoAtNet-d4 multi-scale whole-volume center crop (224x224).
3. **Engine 3 (ConvNeXt R6 Flip TTA)**: ConvNeXt-Small with 3D joint-space localization via SKM-TEA (`AnchorBank`), ResNet-34 side canonicalization, and native $352 \times 352$ joint crops.
4. **Engine 4 (Engine 5 Plane-Decoupled Specialists)**:
   - Slices: 2.5D slice triples $(c-1, c, c+1)$ from the 96-slice volume.
   - Sagittal Specialist: ACL, Medial Meniscus, Lateral Meniscus, Contusion.
   - Coronal Specialist: MCL, Medial OA, Lateral OA.
   - Axial Fluid Specialist: PF OA, Effusion, Synovitis, Baker's, Fracture.
   - Backed by ADR-015 and ADR-020. Eliminates multi-task representation collapse (off-diagonal correlation $0.4490$).

### Blending Scheme
- **Acute Tears (Focal Amplification)**:
  - ACL, Medial Meniscus, Lateral Meniscus, Fracture
  - Generalized Power-Mean ($p = 1.4$):
    $$b = \left(0.40 \cdot r_{\text{raptor}}^{1.4} + 0.40 \cdot r_{\text{convnext}}^{1.4} + 0.20 \cdot r_{\text{specialist}}^{1.4}\right)^{1 / 1.4}$$
- **Diffuse Findings**:
  - Linear Rank Blend:
    $$b = 0.40 \cdot r_{\text{raptor}} + 0.40 \cdot r_{\text{convnext}} + 0.20 \cdot r_{\text{specialist}}$$
  - Synovitis & Effusion benefit directly from Engine 5's dedicated fluid specialist ($0.8917$ and $0.8138$ OOF AUC).

### Offline Verification & Artifacts
- **Kernel Metadata**: `cyruspinto/rsna-knee-v21-quad-engine-blend`
- **Kernel Sources**: `cyruspinto/rsna-knee-engine5-decoupled-train` (Zero bandwidth upload required; mounts directly in Kaggle).
- **Fast DICOM Reader**: SIMD multi-threaded C++ `_fastread.py`.
