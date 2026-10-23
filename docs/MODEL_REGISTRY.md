# RSNA Knee Abnormality Detection: Comprehensive Model Registry (Engines 1 to 14)

Detailed architectural specifications, training configurations, input representations, and validation telemetry across all 14 model engines developed during the competition.

## Master Engine Overview

| Engine ID | Architecture Family | Input Tensor | Pretraining Foundation | Key Innovations | Standalone Gold-58 AUC | Peak Role in System |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| **Engine 1** | ResNet-50 | 2D Single-Slice 224px | ImageNet-1k | Initial baseline prototype | 0.8850 | Deprecated (Phase 1) |
| **Engine 2** | EfficientNet-B4 | 2D Single-Slice 380px | ImageNet-1k | Compound scaling, uncalibrated | 0.9020 | Deprecated (Phase 1) |
| **Engine 3** | DenseNet-121 3D | 3D Volume (32x256x256) | From scratch | Dense volumetric connectivity | 0.9140 | High VRAM; deprecated |
| **Engine 4** | CoAtNet-rmlp-2 384px | 96 Slices 2.5D (384px) | ImageNet-12k + OAI | Depthwise Conv + Global Self-Attn | 0.9420 | Predecessor to Raptor-384 |
| **Engine 5** | Raptor CoAtNet Multi-Scale | 96 Slices (384px + 224px) | ImageNet-12k + OAI | 12 finding-specific attention queries | 0.9520 | **Anchor Stage 1 (65% 384 + 35% 224)** |
| **Engine 6** | ConvNeXt-Large R6 | 64 Slices Compact (384px) | ImageNet-22k + OAI | 100mm bilateral locals + D2 fluid | 0.9490 | **Anchor Stage 2 (Heliosli R6 Bundle)** |
| **Engine 7** | Dual-Engine Rank Blend | Multi-Engine Fusion | OAI-Pretrained | 50/50 Percentile Rank Blend (V6) | 0.9510 | **V6 Breakthrough (0.9560 LB)** |
| **Engine 8** | Generalized Power-Rank | Multi-Scale Power-Mean | OAI-Pretrained | Non-linear exponent p=1.4 on focal tears | 0.9530 | **V20 Private Anchor (0.9420 Private)** |
| **Engine 9** | Sovereign Rank Master | Dual-Arm + Biomechanics | OAI-Pretrained | Physiological coupling + fail-soft | 0.9550 | **V45 Public Peak (0.9580 LB)** |
| **Engine 10** | Swin-Base-384 5-Fold | 16-Window Triplets | ImageNet-22k | 5-Fold shifted window self-attention | 0.8750 | Auxiliary feature diversity |
| **Engine 11** | 2.5D Fast ConvNeXt Reader | 3-Slice Windows | ImageNet-1k | Lightweight fast orthogonal predictions | 0.9120 | **Stage 5 Versia_7 Reader (0.016 mean)** |
| **Engine 12** | ConvNeXt-Large-384 MIL | 16-Triplet Dynamic Stride | ImageNet-22k | Gated Attention MIL | 0.8978 | **Specialist Stage 6 Branch (Medial OA)** |
| **Engine 13** | Swin-Base-384 MIL | 16-Triplet Dynamic Stride | ImageNet-22k | Asymmetric Focal Loss (gamma-=4, gamma+=1) | 0.8577 | **Specialist Stage 6 Branch (Effusion/Fracture)** |
| **Engine 14** | DINOv2 ViT-B/14 Reg4 | Cross-Planar Self-Attn | Self-Supervised LVD-142M | Register tokens + planar cross-attention | 0.8890 | Cross-planar orthogonal embeddings |

---

## Detailed Architectural Profiles

### Engine 5: Raptor CoAtNet Multi-Scale (Stage 1 Anchor)
* **Backbones**: `coatnet_rmlp_2_rw_384` (65% share) and `coatnet_rmlp_2_rw_224` (35% share, 0.80 center crop).
* **Input Tensor**: 5-slot physical volume tensor mapped to 96 slices (Slot 1: Sag FS 26, Slot 2: Sag T1 22, Slot 3: Cor FS 18, Slot 4: Cor T1 12, Slot 5: Axial 18).
* **Attention Mechanism**: 12 finding-specific linear query heads ($W_c \in \mathbb{R}^{256 \times 1}$).
* **TTA**: Plain + horizontally mirrored passes on the 384px backbone.

### Engine 6: Heliosli ConvNeXt-Large R6 (Stage 2 Counterbalance)
* **Backbone**: `convnext_large.fb_in22k_ft_in1k_384`
* **Bundle ID**: `fe055406fad6a12f` (4-fold SWA average)
* **Local Views**: 100mm bilateral compartment localized crops focusing on medial and lateral joint spaces.
* **Morphological Fluid Measurement**: Quantifies fluid volume (mL) from lowest-background sagittal series ($P_{99} \ge 229$).
* **Flip TTA**: Uses anatomical permutation operator $P = (\text{Medial Meniscus} \leftrightarrow \text{Lateral Meniscus}) \times (\text{Medial OA} \leftrightarrow \text{Lateral OA})$.

### Engine 12: ConvNeXt-Large-384 Gated MIL (Stage 6 Specialist)
* **Backbone**: `convnext_large.fb_in22k_ft_in1k_384`
* **Attention Formulation**: $a_k = \frac{\exp(w^T (\tanh(Vh_k) \odot \sigma(Uh_k)))}{\sum_j \exp(w^T (\tanh(Vh_j) \odot \sigma(Uh_j)))}$
* **Gold-58 Specialist Scores**: Medial OA (0.9829), Medial Meniscus (0.9255), ACL (0.9375).

### Engine 13: Swin-Base-384 with Asymmetric Focal Loss (Stage 6 Specialist)
* **Backbone**: `swin_base_patch4_window12_384.ms_in22k_ft_in1k`
* **Loss**: Asymmetric Focal Loss with $\gamma_- = 4, \gamma_+ = 1$, margin $m = 0.05$.
* **Gold-58 Specialist Scores**: Joint Effusion (0.9938), Bone Fracture (0.9278), Bone Contusion (0.9487).
