# RSNA Knee Abnormality Detection: Multi-Engine Volumetric Rank-Fusion System



[Private LB Verified](https://www.google.com/search?q=https://www.kaggle.com/code/cyruspinto/rsna-knee-v20-power-rank-blend%3FscriptVersionId%3D358962681)



Official technical repository, architectural specification, and complete engineering documentation for our solution in the **RSNA Knee Abnormality Detection Competition** hosted by the Radiological Society of North America (RSNA) and the American Society of Skeletal Radiology (ASSR).

This codebase documents our entire competitive trajectory: from raw DICOM physical space alignment and multi-scale attention backbones, through the **0.9580 Public Leaderboard milestones** ([Candidate V20 Power Rank Blend](https://www.google.com/search?q=https://www.kaggle.com/code/cyruspinto/rsna-knee-v20-power-rank-blend%3FscriptVersionId%3D358962681) and [Candidate V45 Sovereign Rank Master](https://www.kaggle.com/code/cyruspinto/rsna-knee-v45-sovereign-rank-master?scriptVersionId=360231579)), to our forensic post-competition audit and the final **Candidate V60 Pure-OAI & Versia Power Master** architecture.

```
flowchart LR
    A[Raw Multi-Plane DICOMs] --> B[Zero-Copy POSIX Reader\n3D Slice Normal Sorting]
    B --> C[140mm Physical Metric Crop\n5-Slot Canonical Tensor]
    C --> D[Raptor-384 CoAtNet\nMirror TTA]
    C --> E[Raptor-224 CoAtNet\nFocal Center Crop]
    C --> F[Heliosli ConvNeXt-L R6\nBilateral Locals + Fluid Vol]
    C --> G[2.5D Fast Reader\nVersia_7 Calibration]
    D & E & F & G --> H[Empirical Percentile\nRank Standardization]
    H --> I[Continuous Power-Probit\nTail Calibration]
    I --> J[12x12 Orthopaedic\nBiomechanical Coupling]
    J --> K[submission.csv\nDual-T4: 451.3s - 469.4s]

```

## Table of Contents

1. [Executive Summary & Key Results](#1-executive-summary--key-results)

2. [Task Dynamics & Musculoskeletal Pathology Targets](#2-task-dynamics--musculoskeletal-pathology-targets)

3. [End-to-End System Architecture](#3-end-to-end-system-architecture)

   * [3.1 Volumetric Normalization & Fast I/O Pipeline](#31-volumetric-normalization--fast-io-pipeline)

   * [3.2 Multi-Scale Raptor CoAtNet Backbone (384px + 224px)](#32-multi-scale-raptor-coatnet-backbone-384px--224px)

   * [3.3 Heliosli ConvNeXt-Large R6 4-Fold Bundle](#33-heliosli-convnext-large-r6-4-fold-bundle)

   * [3.4 2.5D Fast Reader & Specialist MIL Engines](#34-25d-fast-reader--specialist-mil-engines)

4. [Calibration Mathematics & Ensembling Theory](#4-calibration-mathematics--ensembling-theory)

   * [4.1 Empirical Percentile Rank Standardization](#41-empirical-percentile-rank-standardization)

   * [4.2 Generalized Power-Mean Blending (p = 1.02 - 1.40)](#42-generalized-power-mean-blending-p--102---140)

   * [4.3 Probit Normal Transformation (Acklam Rational Approximation)](#43-probit-normal-transformation-acklam-rational-approximation)

   * [4.4 12x12 Orthopaedic Biomechanical Kinetic Coupling Matrix](#44-12x12-orthopaedic-biomechanical-kinetic-coupling-matrix)

5. [Complete Evolutionary Trajectory (V1 to V60)](#5-complete-evolutionary-trajectory-v1-to-v60)

6. [Forensic Post-Mortem & Diagnostic Case Studies](#6-forensic-post-mortem--diagnostic-case-studies)

7. [Repository Structure & Codebase Map](#7-repository-structure--codebase-map)

8. [Reproducibility & Execution Guide](#8-reproducibility--execution-guide)

9. [Credits, Citations & Compliance Acknowledgements](#9-credits-citations--compliance-acknowledgements)

## 1. Executive Summary & Key Results

The challenge requires automated multi-label classification of **12 distinct knee abnormalities** from complete multi-sequence, multi-plane magnetic resonance imaging (MRI) studies. Evaluation is governed by macro-averaged Area Under the Receiver Operating Characteristic Curve (**Macro ROC-AUC**) across all 12 pathologies, with an explicit computational ceiling enforcing full inference in $\le 500\text{ seconds}$ on Dual NVIDIA Tesla T4 GPUs for the **\$18,000 Efficiency Prize**.

| Evaluation Attribute | Verified Value / Specification | Contextual Benchmark | 
| ----- | ----- | ----- | 
| **Peak Public Leaderboard** | **0.9580** | Top tier cluster (Rank #859 tied cohort) | 
| **Official Private Leaderboard (V20)** | **0.9420** | Verified Kaggle Private Evaluation (Version 1 of 1) | 
| **Gold-58 Adjudicated Validation** | **0.9582** | 2,000-iteration paired bootstrap on radiologist ground truth | 
| **Kaggle Kernel Latency (V20)** | **469.4 s** (451.3s internal) | Fully compliant with $500\text{s}$ Efficiency Prize limit | 
| **Kaggle Kernel Latency (V45)** | **533.9 s** | High cold-start I/O (model weights loading: 24s & 36s) | 
| **Candidate V60 Production Latency** | **452.1 s** | Zero triple-quantization, continuous Power-Probit | 
| **Target Findings** | 12 Musculoskeletal Pathologies | Ligament, meniscus, osteoarthritis, fluid, trauma | 
| **Primary Compute Substrates** | Dual Tesla T4 GPUs (15 GB each) | Pinned PyTorch 2.11.0+cu128 offline execution | 

### Key Architectural Invariants

* **Percentile Rank Standardization**: Because macro ROC-AUC depends strictly on within-finding relative ordering, raw sigmoid probabilities from divergent architectures are converted to empirical percentile ranks in $[1/N, 1.0]$, neutralizing calibration compression.

* **3D Normal Spatial Slice Projection**: Slices are ordered physically using the plane normal cross-product $n = \text{IOP}_{0:2} \times \text{IOP}_{3:5}$ and projected depth $d = \text{IPP} \cdot n$. Correcting spatial ordering yielded a **+0.0280 AUC jump** over naive filename lexical sorting.

* **140 mm Metric Physical FOV Normalization**: Standardizes variable scanner pixel spacings ($0.25\text{ mm/px}$ to $0.65\text{ mm/px}$) to a uniform physical bounding box, ensuring the femoral condyles and tibial plateau span identical receptive fields.

* **Zero 16-Window Non-OAI Corpus Dilution**: Restricts auxiliary non-OAI models to surgical, class-specific tie-breaking weights ($0.003$ to $0.035$, mean $0.016$), preserving deep OAI-pretrained representations.

## 2. Task Dynamics & Musculoskeletal Pathology Targets

Each test study comprises multiple MRI series acquired across three anatomical planes (Sagittal, Coronal, Axial) and distinct contrast weightings (Fluid-Sensitive T2/PD/STIR vs Non-Fluid-Sensitive T1). The target space encompasses 12 non-mutually-exclusive binary conditions:

| Finding Index | Clinical Abnormality Target | Anatomical Class | Typical Pathognomonic Views | Baseline Prevalence | 
| ----- | ----- | ----- | ----- | ----- | 
| **0** | **ACL Tear** | Focal Acute Trauma | Sagittal FS / Non-FS (Intercondylar notch) | \~18.5% | 
| **1** | **MCL Tear** | Peripheral Acute | Coronal FS (Medial collateral femoral/tibial) | \~14.2% | 
| **2** | **Medial Meniscus Tear** | Focal Acute / Degenerative | Sagittal + Coronal (Posterior horn bow-ties) | \~28.4% | 
| **3** | **Lateral Meniscus Tear** | Focal Acute / Degenerative | Sagittal + Coronal (Anterior/posterior horns) | \~19.8% | 
| **4** | **Medial Compartment OA** | Diffuse Chronic | Coronal (Cartilage thinning, subchondral sclerosis) | \~31.2% | 
| **5** | **Lateral Compartment OA** | Diffuse Chronic | Coronal (Lateral tibiofemoral joint space narrowing) | \~16.5% | 
| **6** | **Patellofemoral (PF) OA** | Diffuse Chronic | Axial + Sagittal (Retropatellar cartilage loss) | \~22.4% | 
| **7** | **Joint Effusion** | Fluid Distension | Sagittal FS + Axial (Suprapatellar pouch fluid) | \~41.5% | 
| **8** | **Synovitis** | Synovial Inflammation | Sagittal FS + Axial (Synovial fringe thickening) | \~24.5% | 
| **9** | **Baker's Cyst** | Fluid Distension | Sagittal FS + Axial (Gastrocnemius-semimembranosus) | \~11.2% | 
| **10** | **Bone Contusion** | Trabecular Microfracture | Sagittal FS + Coronal FS (Subchondral edema) | \~17.6% | 
| **11** | **Bone Fracture** | Cortical Disruption | Multi-Planar (Cortical breach, impaction step-off) | \~5.2% | 

## 3. End-to-End System Architecture

The production inference pipeline coordinates three independent inference engines in isolated subprocesses to guarantee full GPU VRAM release between stages, followed by continuous rank fusion.

```
flowchart TD
    subgraph INGEST [Raw Study DICOMs]
        D[Test Series DICOMs]
    end

    subgraph PREPROC [Zero-Copy Fast I/O Pipeline]
        P1[study_lut POSIX Reader]
        P2[3D Plane Normal Ordering: n = IOP_row x IOP_col]
        P3[140mm Metric Physical Bounding Box Crop]
        P4[5-Slot Canonical Volumetric Tensor: 96 Slices]
    end

    subgraph ENGINE1 [Stage 1: Multi-Scale Raptor CoAtNet]
        R1[Native 384px: 94 Triplet Windows]
        R2[Focal 224px: 0.8 Center Crop]
        R3[12 Finding-Specific Query Heads: W_c]
        R4[Mirror TTA: Plain + Mirrored Logit Mean]
    end

    subgraph ENGINE2 [Stage 2: Heliosli ConvNeXt-L R6 Bundle]
        H1[4-Fold SWA Checkpoints via probe_dual.py]
        H2[100mm Bilateral Compartment Locals]
        H3[Morphological D2 Fluid Volume Quantifier]
        H4[Flip TTA with Anatomical Permutation Operator P]
    end

    subgraph ENGINE3 [Stage 3: 2.5D Fast Reader]
        RD1[ConvNeXt 3-Slice Windows]
        RD2[Calibrated Versia_7 Weights: 0.003 - 0.035]
    end

    subgraph FUSION [Stage 4: Continuous Power-Probit Fusion]
        F1[Empirical Percentile Rank Standardization: RankPct in 1/N, 1.0]
        F2[Generalized Power-Mean: p = 1.02 - 1.40 on Focal Lesions]
        F3[Acklam Probit Normal Transformation: z = Phi^-1 of r]
        F4[12x12 Orthopaedic Biomechanical Coupling Matrix]
        F5[Fail-Soft Boundary Clamping: 0.0001, 0.9999]
    end

    D --> P1 --> P2 --> P3 --> P4
    P4 --> R1 & R2 --> R3 --> R4
    P4 --> H1 --> H2 & H3 --> H4
    P4 --> RD1 --> RD2
    R4 & H4 & RD2 --> F1 --> F2 --> F3 --> F4 --> F5 --> OUT[submission.csv: N x 13]

```

### 3.1 Volumetric Normalization & Fast I/O Pipeline

To resolve scanner heterogeneity across international imaging centers, all incoming DICOM series are mapped to a canonical 5-slot physical volume tensor:

$$
\mathbf{V}_{\text{study}} \in \mathbb{R}^{96 \times 384 \times 384}
$$

```
+------------------------------------------------------------------------------------+
| 5-SLOT PROTOCOL SPECIFICATION (Total = 96 Slices Native / 64 Slices Compact)       |
+------------------------------------------------------------------------------------+
| Slot 1: Sagittal Fluid-Sensitive (FS)      | 26 slices (native) | 18 slices (comp) |
| Slot 2: Sagittal Non-Fluid-Sensitive (T1)  | 22 slices (native) | 14 slices (comp) |
| Slot 3: Coronal Fluid-Sensitive (FS)       | 18 slices (native) | 12 slices (comp) |
| Slot 4: Coronal Non-Fluid-Sensitive        | 12 slices (native) |  8 slices (comp) |
| Slot 5: Axial (All Modalities)             | 18 slices (native) | 12 slices (comp) |
+------------------------------------------------------------------------------------+

```

1. **3D Plane Normal Ordering**: Evaluates directional cosines from tag `(0028, 0030) ImageOrientationPatient` $[r_x, r_y, r_z, c_x, c_y, c_z]$. Slices are sorted along the slice normal $n = r \times c$ via depth projection $d = \text{IPP} \cdot n$:
   

   $$
   n = r \times c = \begin{bmatrix} r_y c_z - r_z c_y \\ r_z c_x - r_x c_z \\ r_x c_y - r_y c_x \end{bmatrix}, \quad d_i = \mathbf{p}_i \cdot n
   $$

2. **Metric Physical Cropping**: Uses in-plane spacing $(s_y, s_x)$ from tag `(0028, 0030) PixelSpacing`:
   

   $$
   \text{CropPixels}_y = \text{round}\left(\frac{140.0\text{ mm}}{\max(s_y, 10^{-3})}\right), \quad \text{CropPixels}_x = \text{round}\left(\frac{140.0\text{ mm}}{\max(s_x, 10^{-3})}\right)
   $$

3. **Wide-Span Quantile Slicing (**$2\%\text{ to }98\%$**)**: Samples linearly across $[0.02 \cdot N_s, 0.98 \cdot N_s - 1]$, capturing collateral ligaments and peripheral meniscal horns.

4. **2.5D Triplet Windowing**: Stacks 3 physically adjacent slices into RGB-channel inputs $(z-1, z, z+1)$, yielding $K = 94$ overlapping evaluation windows.

### 3.2 Multi-Scale Raptor CoAtNet Backbone (384px + 224px)

The visual foundation is anchored by two CoAtNet models (`coatnet_rmlp_2_rw_384` and `coatnet_rmlp_2_rw_224`) pretrained on ImageNet-12k and fine-tuned on the expanded OAI-augmented knee corpus:

* **Scale 1 (Primary, 65% Raptor Weight)**: Full $140\text{ mm}$ bounding box resampled to $384 \times 384$.

* **Scale 2 (Focal Context, 35% Raptor Weight)**: Centered $0.80$ crop of the $140\text{ mm}$ field resampled to $224 \times 224$.

* **Per-Finding Attention Head**: Avoids uniform global pooling by assigning an independent projection query $W_c \in \mathbb{R}^{256 \times 1}$ to each pathology:
  

  $$
  a_{k, c} = \frac{\exp\left(W_c^T \tanh(V h_k)\right)}{\sum_{j=1}^K \exp\left(W_c^T \tanh(V h_j)\right)}, \quad z_c = \sum_{k=1}^K a_{k, c} h_k \cdot \text{clsW}_c + \text{clsb}_c
  $$

  
  This allows focal acute pathologies (ACL tears localized to 2 sagittal slices) to attend to their pathognomonic slices, while diffuse conditions (Osteoarthritis) aggregate across coronal cartilage compartments.

* **Mirror Test-Time Augmentation (TTA)**: Executes a plain forward pass and a horizontally mirrored forward pass on the 384px model, averaging probabilities in logit space.

### 3.3 Heliosli ConvNeXt-Large R6 4-Fold Bundle

The second engine provides orthogonal architectural inductive bias via `convnext_large.fb_in22k_ft_in1k_384` managed by `probe_dual.py`:

* **Bilateral Coronal & Centered Locals**: Generates localized $100\text{ mm}$ bounding crops focusing specifically on the medial and lateral joint compartments.

* **Morphological D2 Fluid Volumetry**: Extracts the lowest-background fluid-sensitive sagittal series, applies thresholding at the 99th intensity percentile ($P_{99} \ge 229$), executes morphological opening with an elliptical structuring element ($r = 1.5\text{ mm} / s_x$), and quantifies absolute fluid volume in milliliters:
  

  $$
  V_{\text{fluid}} = \sum_{z=1}^{N_z} \left(\sum_{x, y} M_{\text{open}}(x, y, z) \cdot s_x s_y\right) \cdot \frac{\Delta z}{1000} \quad [\text{mL}]
  $$

* **Anatomical Permutation Flip TTA**: Evaluates native and mirrored volumes, re-indexing symmetric bilateral findings with permutation operator $P$:
  

  $$
  P = (\text{Medial Meniscus} \leftrightarrow \text{Lateral Meniscus}) \times (\text{Medial OA} \leftrightarrow \text{Lateral OA})
  $$

### 3.4 2.5D Fast Reader & Specialist MIL Engines

A lightweight 2.5D ConvNeXt reader (`_reader_sub.csv`) provides additional feature diversity. In accordance with the Versia_7 benchmark, this engine is blended with **conservative, finding-specific weights** ($0.003$ to $0.035$, mean $0.016$) rather than uniform blending ($0.15$), preventing dilution of the deep OAI-pretrained representations.

## 4. Calibration Mathematics & Ensembling Theory

### 4.1 Empirical Percentile Rank Standardization

Because the evaluation metric is Macro Mean ROC-AUC, absolute probability calibration is irrelevant; only relative ordering within each pathology column determines performance. Direct linear averaging of raw sigmoid outputs $\sigma(z_1)$ and $\sigma(z_2)$ creates severe distortion because the steeper model dominates the rank ordering.

We map raw model scores to empirical percentiles over the test cohort:

$$
\text{RankPct}(x_i) = \frac{\text{rank}(x_i)}{N} \in [1/N, 1.0]
$$

| Finding Target | Raptor Share ($w_{\text{rap}}$) | Heliosli Share ($w_{\text{hel}}$) | Versia_7 Reader ($w_{\text{rdr}}$) | Power Exponent ($p$) | 
| ----- | ----- | ----- | ----- | ----- | 
| **ACL** | 0.450 | 0.550 | 0.025 | 1.04 (1.4 in V20) | 
| **MCL** | 0.435 | 0.565 | 0.020 | 1.05 (1.0 in V20) | 
| **Medial Meniscus** | 0.450 | 0.550 | 0.020 | 1.02 (1.4 in V20) | 
| **Lateral Meniscus** | 0.450 | 0.550 | 0.008 | 1.03 (1.4 in V20) | 
| **Medial OA** | 0.450 | 0.550 | 0.025 | 1.00 (1.0 in V20) | 
| **Lateral OA** | 0.195 | 0.805 | 0.008 | 1.02 (1.0 in V20) | 
| **PF OA** | 0.450 | 0.550 | 0.008 | 1.02 (1.0 in V20) | 
| **Effusion** | 0.445 | 0.555 | 0.004 | 1.00 (1.0 in V20) | 
| **Synovitis** | 0.735 | 0.265 | 0.003 | 1.02 (1.0 in V20) | 
| **Baker's Cyst** | 0.450 | 0.550 | 0.035 | 1.06 (1.0 in V20) | 
| **Bone Contusion** | 0.485 | 0.515 | 0.030 | 1.04 (1.0 in V20) | 
| **Bone Fracture** | 0.485 | 0.515 | 0.008 | 1.06 (1.4 in V20) | 

### 4.2 Generalized Power-Mean Blending (p = 1.02 - 1.40)

Linear rank blending enforces equal marginal importance across the entire interval $[0, 1]$. For acute focal lesions (ACL, Meniscus, Fracture), high-confidence concordant detections between independent backbones represent verified pathology that should be amplified non-linearly without artificial heuristic thresholding.

We formulate the Generalized Power-Mean:

$$
M_p(x, y; w) = \left( w \cdot x^p + (1 - w) \cdot y^p \right)^{1/p}
$$

* **Focal Acute Pathologies (**$p = 1.02\text{ to }1.06$**,** $p=1.40$ **in V20)**: Amplifies high-percentile signals smoothly without heuristic threshold clipping.

* **Diffuse Degenerative Pathologies (**$p = 1.00$**)**: Preserves linear rank blending on Osteoarthritis and Effusion.

### 4.3 Probit Normal Transformation (Acklam Rational Approximation)

Uniform percentile ranks compress the extreme tails: the numerical difference between the 99th and 99.9th percentile is identical to the difference between the 50th and 50.9th percentile ($0.009$). In diagnostic radiology, extreme upper-tail conviction must be preserved.

Using Acklam's rational approximation for the Inverse Normal Cumulative Distribution Function $\Phi^{-1}(p)$, uniform ranks are mapped to Gaussian $z$-scores:

$$
z_i = \Phi^{-1}\left( \text{clip}\left( r_i \cdot \frac{N - 1}{N} + \frac{0.5}{N}, 10^{-4}, 1 - 10^{-4} \right) \right)
$$

Ensembling is performed in Gaussian space and projected back via $\Phi(z)$, enabling extreme tail predictions to resolve near-ties cleanly:

$$
z_{\text{blend}} = \sum_{m} w_m \Phi^{-1}(r_m), \quad \hat{s}_{\text{probit}} = \Phi(z_{\text{blend}}) = \frac{1}{2} \left[ 1 + \text{erf}\left(\frac{z_{\text{blend}}}{\sqrt{2}}\right) \right]
$$

### 4.4 12x12 Orthopaedic Biomechanical Kinetic Coupling Matrix

Musculoskeletal knee injuries follow well-established kinetic trauma pathways:

```
[Acute Trauma Kinetic Cascade]
  High-Energy Pivot-Shift Mechanism
    ├──> Complete ACL Rupture (Col 0)
    │     ├──> Lateral Condyle Trabecular Contusion (+3.0% coupling to Col 10)
    │     └──> Lateral Meniscal Root / Posterior Tear (+2.0% coupling to Col 3)
    └──> Capsular Hemarthrosis / High-Volume Joint Effusion (Col 7)

[Chronic Degenerative Cascade]
  Medial Compartment Overload
    ├──> Medial Tibiofemoral Chondromalacia / Medial OA (Col 4)
    │     └──> Medial Meniscal Extrusion & Maceration (+1.5% bidirectional to Col 2)
    └──> Tricompartmental Progression -> Patellofemoral (PF) OA (+1.2% to Col 6)

[Hydrostatic Fluid Pressure Cascade]
  Intra-Articular Hydrops / Effusion (Col 7)
    ├──> Elevated Hydrostatic Pressure -> Reactive Synovitis (+1.5% bidirectional to Col 8)
    ├──> Gastrocnemius-Semimembranosus Valve Extravasation -> Baker's Cyst (+1.5% to Col 9)
    └──> Subchondral Bone Compromise -> Trabecular Fracture (+1.5% to Col 11)

```

The mathematical update rule is evaluated as:

$$
R_{c}^{\text{final}} = \left(1 - \sum_{k \neq c} \lambda_{c, k}\right) R_c + \sum_{k \neq c} \lambda_{c, k} R_k
$$

This linear formulation operates directly on continuous percentile scores, preventing discrete step quantization.

## 5. Complete Evolutionary Trajectory (V1 to V60)

| Phase / Release | Key Technical Innovations & Strategy | Public LB | Private LB | Gold-58 Gate | Dual-T4 Latency | Efficiency Status | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Phase 0: Foundation** | Qwen-2.5 72B weak label extraction from 4,349 reports; Gold-58 gate | N/A | N/A | 0.9172 CV | N/A | Solidified labels | 
| **V1 - V5: Prototypes** | Naive 2D backbones; pydicom loading; uncalibrated sigmoids | 0.916 - 0.938 | — | 0.8950 | 510.0 s | Filename sorting flaw | 
| **V6: Dual-Engine** | C++ `_fastread.py` 3D-normal sorting; 140mm crop; 50/50 rank blend | 0.9560 | — | 0.9420 | 480.0 s | **+0.0280 AUC breakthrough** | 
| **V7: Asymmetric** | Class-dependent 65/35 and 70/30 linear rank blend | 0.9550 | — | 0.9410 | 485.0 s | Small sample overfitting | 
| **V8: Direct Flip TTA** | Unpaired horizontal flip test-time augmentation | 0.9550 | — | 0.9405 | 490.0 s | Inverted lateral/medial | 
| **V9 - V11: Heuristics** | Rule-based conditional priors (e.g. if ACL then Contusion floor) | 0.9550 | — | 0.9415 | 485.0 s | Step discontinuities | 
| **V12: Clean Gate** | Excised heuristic rules (ADR-001/002); paired bootstrap gate | 0.9560 | — | 0.9430 | 475.0 s | Pure rank baseline restored | 
| **V13 - V16: Multi-Scale** | Integrated 224px CoAtNet companion (0.8 center crop) | 0.9570 | — | 0.9480 | 460.0 s | Refined meniscal signal | 
| **V17: Personal Best** | Full multi-scale Raptor (384 + 224) + Heliosli R6 frozen bundle | **0.9580** | — | 0.9520 | 443.4 s | **Personal Best public benchmark** | 
| **V18 - V19: Tuning** | Per-finding weight sweeps (Lateral OA 0.20, Synovitis 0.80) | 0.9580 | — | 0.9525 | 450.0 s | Stabilized specialist weights | 
| **V20: Power Rank Blend** | Generalized Power-Mean ($p=1.40$) on ACL, Menisci, Fracture | **0.9580** | **0.9420** | 0.9530 | **469.4 s** (451.3s int) | **Official Verified Private LB Anchor** | 
| **V45: Sovereign Master** | Dual-arm anchor + Biomechanical coupling + Fail-soft clamping | **0.9580** | — | 0.9550 | 533.9 s | High cold-start weights I/O | 
| **V46 - V48: Dilution** | Auxiliary 16-window non-OAI models (`all_vols.npy`) blended at 20-50% | 0.953 - 0.957 | — | 0.9480 | 520.0 s | Non-OAI feature dilution | 
| **V50: Automation Outage** | Script queried hidden test UIDs in training corpus `all_vols.npy` | **0.4910** | — | 0.7200 | 279.8 s | Fallback to random beta noise | 
| **V51 - V52: Streaming** | Live DICOM streaming + 0.01s path resolver; zero random noise | 0.9560 | — | 0.9530 | 335.0 s | Restored streaming baseline | 
| **V53: Real DICOM Apex** | Asynchronous Dual-T4 dispatch; 3-point monotonic slice check | 0.9570 | — | 0.9540 | 510.0 s | Triple rank-quantization identified | 
| **V56: Sovereign Reader** | Added 2.5D ConvNeXt Reader; Reader weighted at $W=0.15$ | 0.9570 | — | 0.9540 | 551.7 s | Slight dilution from $W=0.15$ | 
| **V58: Worker Thrashing** | Raised thread pool from 6 to 12 workers on 4 vCPUs | 0.9570 | — | 0.9540 | 660.9 s (+109s) | Severe GIL & disk cache thrashing | 
| **V59: Probit Grandmaster** | Hybrid 50% Gaussian Probit $\Phi^{-1}(p)$ + 50% Continuous Rank | 0.9580 | — | 0.9565 | 530.0 s | Continuous margin preservation | 
| **V60: Pure-OAI Master** | Clean V45 + exact Versia_7 reader (0.003-0.035) + 12x12 coupling | **0.9585+** | — | **0.9582** | **452.1 s** | **Final Production Architecture** | 

## 6. Forensic Post-Mortem & Diagnostic Case Studies

### Case 1: The V50 Zero-Score Catastrophe (LB 0.4910)

* **Root Cause**: An automated script runner replaced the production DICOM pipeline with an offline lookup inside `all_vols.npy`. However, `all_vols.npy` contains strictly the 4,407 training studies. When executed against hidden test studies, lookup queries returned empty arrays, triggering fail-soft fallback code that populated predictions with `np.random.beta(0.5, 2.0)` noise.

* **Engineering Remedy**: Hidden test studies must stream and decode directly from `test_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm`. The AST Pre-Flight Guardrail now explicitly prohibits training corpus lookups on evaluation studies.

### Case 2: The Triple Rank-Quantization Cliff (0.9580 to 0.9570)

* **Root Cause**: In Candidate V53/V56, `rank_matrix()` was invoked sequentially across three stages: once after base anchor generation, once after auxiliary branch blending, and once inside `apply_biomechanical_coupling()`. On small batches, ranking snaps smooth continuous predictions to discrete fractions ($0.3333, 0.6667, 1.0000$), artificially amplifying subtle $0.0333$ margin differences into massive $0.3333$ cliffs.

* **Engineering Remedy**: Replaced discrete step ranking with continuous convex blending and Probit Gaussian tail calibration ($\Phi^{-1}(p)$), preserving exact continuous margin separations.

### Case 3: Auxiliary 16-Window Feature Dilution (0.9580 to 0.9530)

* **Root Cause**: Ensembling 16-window non-OAI auxiliary models (`all_vols.npy`) at $20\%\text{ to }50\%$ weights steadily eroded leaderboard scores ($50\% \to 0.953, 35\% \to 0.954, 20\% \to 0.955, 0\% \to 0.958$). The 16-window models lacked the through-plane spatial resolution of 528-window localized OAI models.

* **Engineering Remedy**: Prioritized 528-window OAI models; restricted auxiliary branches to conservative tie-breaking weights ($0.003$ to $0.035$, mean $0.016$) on verified high-performing targets.

### Case 4: CPU Thread Contention in 4-vCPU Virtual Machines (V58)

* **Root Cause**: Raising `ThreadPoolExecutor` from 6 to 12 workers slowed execution by $38\%$ ($660.9\text{ s}$ vs $551.7\text{ s}$) due to GIL thrashing and thread context switching on Kaggle's 4-vCPU allocation.

* **Engineering Remedy**: Locked thread pool concurrency strictly to 6 workers for all asynchronous I/O and preprocessing tasks.

## 7. Repository Structure & Codebase Map

```
rsna-knee-26/
├── README.md                                  # Complete technical documentation & architecture
├── CHANGELOG.md                               # Chronological version & score log
├── LICENSE                                    # MIT License
├── requirements.txt                           # Pinned dependencies (torch 2.11, timm 1.0.25, pydicom 3.0.1)
│
├── assets/                                    # Publication-grade visual figures
│   ├── figure1_evolution_trajectory.png       # Score & latency evolutionary frontier
│   ├── figure2_specialist_weights.png         # Per-pathology weights & power exponents
│   ├── figure3_power_probit_calibration.png   # Power-mean response & Probit quantiles
│   ├── figure4_orthopaedic_coupling.png       # 12x12 anatomical kinetic coupling heatmap
│   └── figure5_pipeline_architecture.png      # End-to-end multi-engine system blueprint
│
├── docs/                                      # Specialized engineering specifications
│   ├── ROADMAP.md                             # Detailed engineering roadmap (V1 to V60)
│   ├── ARCHITECTURE.md                        # Volumetric tensor, I/O & backbone design
│   ├── BLENDING.md                            # Mathematical formulations & calibration proofs
│   ├── CREDITS_AND_COMPLIANCE.md              # OAI/NDA agreements & community credits
│   └── FORENSIC_POSTMORTEM.md                 # Failure mode dissections & profiling logs
│
├── submissions/                               # Production Kaggle submission candidates
│   ├── v20_power_rank_blend/
│   │   ├── rsna-knee-v20-power-rank-blend.ipynb
│   │   └── README.md                          # V20 execution report (0.942 Private, 469.4s)
│   ├── v45_sovereign_rank_master/
│   │   ├── rsna-knee-v45-sovereign-rank-master.ipynb
│   │   └── README.md                          # V45 execution report (0.958 Public, 533.9s)
│   └── v60_pure_oai_master/
│       ├── candidate_v60_pure_oai_versia_power_master.py
│       └── README.md                          # V60 production release specification
│
├── src/                                       # Core modular library
│   ├── fastread/
│   │   ├── study_lut.py                       # Zero-copy POSIX LUT DICOM decoder
│   │   └── geometry.py                        # LPS normal projection & 140mm metric cropping
│   ├── models/
│   │   ├── coatnet_mil.py                     # Raptor CoAtNet with finding-specific attention
│   │   ├── convnext_r6.py                     # Heliosli ConvNeXt-Large with spatial attention
│   │   └── universal_mil.py                   # Exact-match architecture for Engine 12/13
│   ├── volumetry/
│   │   └── fluid_measure.py                   # Morphological D2 fluid volume quantifier
│   └── ensembling/
│       ├── power_probit.py                    # Generalized Power-Mean & Probit transform
│       └── biomechanical_coupling.py          # 12x12 orthopaedic kinetic coupling matrix
│
└── tools/                                     # Verification & automation utilities
    ├── validate_kernel_guardrail.py           # Pre-flight AST static analysis & simulation
    ├── soup_weights.py                        # Weight-Space Model Soup & FP16 compressor
    └── nelder_mead_rank_optimizer.py          # Simplex rank optimizer on Gold-58 gate

```

## 8. Reproducibility & Execution Guide

### Hardware & Environment Requirements

* **Platform**: Kaggle Notebook or Local VM with Dual NVIDIA GPUs ($\ge 15\text{ GB}$ VRAM each).

* **Internet**: Disabled during inference (all wheels, packages, and weights mounted locally).

* **Software**: Python 3.10+, PyTorch 2.11.0+cu128, CUDA 12.8 driver.

### Pinned Checkpoints & Dataset Inputs

| Purpose | Attached Dataset / Checkpoint Slug | SHA-256 Digest / Integrity Verification | 
| ----- | ----- | ----- | 
| **Competition Data** | `rsna-knee-abnormality-detection` | Contains `test.csv`, `test_series.csv`, `test_series/` | 
| **Raptor-384 Weights** | `nartaa/rsna-knee-publication-swa-weights-20261007` | `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef` | 
| **Raptor-224 Weights** | `nartaa/rsna-knee-raptor-alldata-d96-blendjev-oai-c08r224` | `3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6` | 
| **Heliosli Mainline** | `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | 8 module SHA-256 hashes asserted | 
| **Heliosli R6 Bundle** | `heliosli/rsna-knee-r6-frozen-fe055406-w1` | `bundle_id == fe055406fad6a12f` | 
| **Pinned Wheels** | Pydicom 3.0.1, TIMM 1.0.25 | Installed offline from attached wheel files | 

### Running Production Inference Locally or in Kaggle

```
# 1. Clone repository
git clone https://github.com/CYRUS-pinto/rsna-knee-26.git
cd rsna-knee-26

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Run pre-flight AST guardrail verifier
python tools/validate_kernel_guardrail.py --script submissions/v60_pure_oai_master/candidate_v60_pure_oai_versia_power_master.py

# 4. Execute full Candidate V60 production pipeline
python submissions/v60_pure_oai_master/candidate_v60_pure_oai_versia_power_master.py \
    --data-dir /path/to/rsna-knee-abnormality-detection \
    --output-dir ./working \
    --device cuda

```

## 9. Credits, Citations & Compliance Acknowledgements

### Community Acknowledgements

This engineering system builds upon open contributions from the Kaggle competitive imaging community:

* **dreaddevelopment**: For the original Raptor CoAtNet / Multiple Instance Learning inference recipe and baseline weights.

* **nartaa**: For publishing verified high-performing SWA Raptor-384 and Raptor-224 model weights.

* **heliosli**: For the compact-bundle inference architecture, ConvNeXt R6 4-fold bundle, and bilateral local view extraction.

* **varshithakolupuri**: For the multi-scale 224px CoAtNet integration and blend share optimization.

* **goodpjw2008 & evgendvorkin**: For the [RSNA Versia_7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7) multi-model integration insights and calibrated reader weights.

* **stevenleehans, pilkwang & riadmohamed42**: For open-source report-derived weak label sets.

### Controlled-Access OAI / NDA Statement

Data and research tools analyzed in this project were obtained from controlled-access datasets distributed by the **Osteoarthritis Initiative (OAI)** housed within the **NIMH Data Archive (NDA)**:

> *"Data and/or research tools used in the preparation of this manuscript were obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative (OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative informatics system created by the National Institute of Mental Health and the National Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide resource to quicken the pace of biomarker identification, scientific investigation and OA drug development. Dataset identifier: [DOI: 10.15154/0hcg-f676](https://www.google.com/search?q=https://doi.org/10.15154/0hcg-f676)."*

Shared NDA Study: [Study ID 3407](https://www.google.com/search?q=https://nda.nih.gov/study.html%3Fid%3D3407) | [NDA Manuscript Preparation Guidelines](https://nda.nih.gov/nda/manuscript-preparation)