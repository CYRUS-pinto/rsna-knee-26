# RSNA Knee Abnormality Detection: 6-Layer Musculoskeletal Feature Hierarchy

Technical specification of the progressive 6-layer representation hierarchy transforming raw DICOM attenuation voxels into clinically grounded diagnostic probabilities.

```
+-----------------------------------------------------------------------------------+
|               THE 6-LAYER MUSCULOSKELETAL FEATURE HIERARCHY                       |
+-----------------------------------------------------------------------------------+
| Layer 6: 12x12 Orthopaedic Biomechanical Kinetic Coupling Matrix                  |
|          (Trauma cascades, synovial hydrops extravasation, compartment wear)     |
+-----------------------------------------------------------------------------------+
| Layer 5: Continuous Generalized Power-Probit Quantile Calibration                 |
|          (Acklam Phi^-1 normal expansion, power-mean p=1.02-1.40)                 |
+-----------------------------------------------------------------------------------+
| Layer 4: Finding-Specific Attention Query Heads & Gated MIL Aggregation          |
|          (12 independent query heads W_c, through-plane localized pooling)        |
+-----------------------------------------------------------------------------------+
| Layer 3: Multi-Architecture Inductive Foundation (CoAtNet + ConvNeXt + Swin)      |
|          (OAI-pretrained deep backbones, bilateral 100mm locals, D2 volumetry)    |
+-----------------------------------------------------------------------------------+
| Layer 2: 2.5D Sliding Triplet Windowing & Spatial Slice Progression               |
|          (K=94 RGB-stacked triplets (z-1, z, z+1), 3-point monotonic checks)      |
+-----------------------------------------------------------------------------------+
| Layer 1: Physical Coordinate Geometry & 140mm Metric Bounding Box Normalization   |
|          (3D normal projection n=rxc, depth d=IPP.n, PixelSpacing metric scaling) |
+-----------------------------------------------------------------------------------+
```

---

## Layer Engineering Specifications

* **Layer 1 (Physical Coordinate Geometry)**: Directional cosines cross-product $n = \text{IOP}_{0:2} \times \text{IOP}_{3:5}$, depth projection $d = \text{IPP} \cdot n$, and metric pixel crop $\text{round}(140.0\text{ mm} / \max(s_x, 10^{-3}))$.
* **Layer 2 (2.5D Sliding Windowing)**: Stacks adjacent slices into 3-channel RGB triplets $(z-1, z, z+1)$, yielding $K = 94$ continuous sliding evaluation windows across the 96-slice study stack.
* **Layer 3 (Multi-Architecture Foundation)**: CoAtNet-rmlp-2 (depthwise + global attention), ConvNeXt-Large (large $7 \times 7$ kernels + bilateral local crops), and Swin-Base (shifted-window self-attention).
* **Layer 4 (Finding-Specific Attention Query Heads)**: 12 finding-specific projection vectors $W_c \in \mathbb{R}^{256 \times 1}$ allowing focal pathologies to attend strictly to pathognomonic frames.
* **Layer 5 (Power-Probit Quantile Calibration)**: Percentile rank normalization, Generalized Power-Mean ($p=1.02\text{--}1.40$), and Acklam Probit Gaussian quantile projection ($\Phi^{-1}$).
* **Layer 6 (12x12 Orthopaedic Kinetic Coupling)**: Enforces clinical trauma pathways (ACL $\to$ Lateral Contusion & Meniscus; Effusion $\leftrightarrow$ Synovitis; Effusion + Meniscus $\to$ Baker's Cyst; Medial OA $\leftrightarrow$ Medial Meniscus).
