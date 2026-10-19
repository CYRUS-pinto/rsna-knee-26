# ADR-044: First-Principles Scientific Deep-Dive: Computer Vision, Uncertainty-Driven Active Learning & Rank-1 Convergence Blueprint

**Date:** 2026-10-19  
**Status:** ACCEPTED & IN EXECUTION  
**Target:** RSNA Knee Abnormality Detection Challenge  
**Foundations:** Computer Vision Theory, Epipolar & Spatial Geometry, Uncertainty-Driven Active Learning (AL), Semi-Supervised Learning (SSL), and Multi-Task Bayesian Decision Theory.

---

## 1. Executive Summary & Mathematical Foundations

Reaching Rank 1 ($> 0.969\text{ AUC}$) requires transcending ad-hoc heuristic blending and grounding our engineering in **rigorous radiological computer vision and decision theory**.

```
                         Knee Joint MRI Space
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
    Sagittal Plane          Coronal Plane            Axial Plane
   (ACL, PCL, Menisci)     (MCL, LCL, Medial/     (Patellofemoral,
                            Lateral Cartilage)      Effusion, Cyst)
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  ▼
                3D Anatomical Spatial Context
        (Slice Thickness, Epipolar Continuity, Dynamic Stride)
                                  ▼
               Multi-Scale Feature Extraction
       ┌──────────────────────────┴──────────────────────────┐
       ▼                                                     ▼
2.5D Inductive Bias                                  Spatial Vision Transformer
(ConvNeXt-Base/Large: Local texture,                 (Swin-Base: Global compartmental
 cartilage erosion, bone bruises)                     attention & joint space loss)
       │                                                     │
       └──────────────────────────┬──────────────────────────┘
                                  ▼
                   Bayesian Multi-Task Stacking
          (Scale-Invariant Percentile Quantile Calibration)
                                  ▼
            True Positive Rate (TPR) vs False Positive Rate (FPR)
                Optimal Decision Frontier (Macro-AUC Maximized)
```

---

## 2. Deep Dive: Computer Vision & Radiological Geometry for Knee MRI

### A. Anatomical Plane Specialization & Epipolar Slice Alignment
Knee MRIs are acquired in three orthogonal planes, each optimized for distinct musculoskeletal pathologies:
1. **Sagittal Plane:**
   - **Pathologies:** ACL (Anterior Cruciate Ligament), PCL, Meniscal Tears (Medial & Lateral).
   - **Physics:** Slices cut along the anteroposterior axis. Normal ACL appears as a continuous, taut, dark band; tear presents as fiber discontinuity, abnormal orientation angle ($< 45^\circ$ to tibial plateau), or hyperintense fluid signal on T2/PD.
   - **Vision Primitive:** 1D edge orientation tracking along the ligament trajectory across contiguous slices.
2. **Coronal Plane:**
   - **Pathologies:** MCL (Medial Collateral Ligament), LCL, Joint Space Narrowing (Medial OA, Lateral OA).
   - **Physics:** Cuts through the femoral condyles and tibial plateau. Cartilage thinning and osteophyte formation appear symmetrically or asymmetrically in the medial/lateral joint spaces.
   - **Vision Primitive:** Bilateral height difference measurement between femoral condyle and tibial plateau.
3. **Axial Plane:**
   - **Pathologies:** Patellofemoral (PF OA), Joint Effusion, Synovitis, Baker's Cyst.
   - **Physics:** Transverse view of the patella articulating with the femoral trochlea. Excess fluid accumulates in the suprapatellar bursa and popliteal fossa (Baker's cyst).
   - **Vision Primitive:** Morphological connected-component analysis and fluid volume thresholding on T2-weighted fluid-sensitive sequences.

### B. Multi-Scale Receptive Fields: High-Resolution vs. Macro-Geometry
- **Cartilage & Meniscus Defects:** High-frequency, sub-millimeter features requiring $384\times 384$ or $512\times 512$ inputs with small convolution kernels (e.g. $7\times 7$ depthwise in ConvNeXt).
- **Joint Alignment & Compartmental OA:** Low-frequency spatial relationships requiring hierarchical self-attention (Swin-Base windowed attention across 12-slice sequences).

---

## 3. Semi-Supervised Learning (SSL) & Active Learning (AL) on Unlabeled RSNA Studies

The RSNA Knee dataset has a unique structural property:
- Only a **small subset of studies carry gold-standard per-condition labels**.
- The majority of training studies contain **unstructured, multi-lingual radiology reports**.

### A. Uncertainty Sampling & Diversity Sampling
To extract maximum signal without label corruption:
1. **Uncertainty Sampling:**
   $$\text{Uncertainty}(x) = 1 - \max_{y} P(y \mid x)$$
   Studies where ensemble model predictions exhibit maximum entropy ($P \approx 0.50$) represent borderline pathologies (e.g., subtle meniscal fraying vs. complete tear).
2. **Diversity Sampling:**
   Cluster high-dimensional feature embeddings extracted by the pre-trained Swin/ConvNeXt backbones via k-means. Select centroid representatives to ensure scanner, field strength (1.5T vs. 3.0T), and institutional protocol diversity.

### B. Pseudo-Labeling with Temperature Scaling & Asymmetric Loss
- Instead of hard pseudo-labels ($0$ or $1$) which propagate confirmation bias, apply **soft pseudo-labels calibrated with Platt Scaling**:
  $$\hat{y}_i = \sigma\left(\frac{z_i}{T}\right), \quad T > 1.0$$
- Train with **Asymmetric Loss (ASL)** to down-weight easy negative examples and focus learning gradients on rare positive injuries (such as Fractures and MCL tears).

---

## 4. Evaluation Methodology: Dissecting TPR, FPR, PPV, and Macro-AUC

The competition metric is **Macro-Averaged AUC ROC across 12 pathologies**:
$$\text{Macro-AUC} = \frac{1}{12} \sum_{i=1}^{12} \text{AUC}_i$$

### Key Insights:
1. **Rank Invariance:**
   $\text{AUC}_i$ depends **solely on the pairwise ranking order** of positive vs. negative cases:
   $$\text{AUC}_i = \frac{1}{N^+ N^-} \sum_{x^+ \in \mathcal{P}} \sum_{x^- \in \mathcal{N}} \mathbb{I}\left(f(x^+) > f(x^-)\right)$$
   Monotonic transformations (such as uniform percentile ranking) preserve intra-model ranking while standardizing distributions across disparate model architectures.
2. **Threshold Independence:**
   Optimizing for classification accuracy (ACC) or Positive Predictive Value (PPV) using an arbitrary $0.5$ threshold harms AUC! The model should output well-spread, discriminative continuous confidences that separate the positive distribution tail from the negative bulk.

---

## 5. The Top-1 Roadmap: Tasks & Execution Protocol

| Phase | Scientific Task | Objective | Expected Gain |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **V37 Sovereign Titan Master Scoring** | Complete hidden-test execution of the fail-soft 7-engine ensemble. | Validates elimination of assertion crashes & sets new sovereign baseline ($> 0.958$). |
| **Phase 2** | **Targeted Colab A100 Strike** | Train Fold 2 & Fold 3 of Engine 11 ConvNeXt-Base on high-confidence soft report labels with Asymmetric Focal Loss. | Completes 5-fold cross-validation, reducing out-of-sample variance ($\Delta \approx +0.003$). |
| **Phase 3** | **Mathematical OOF Rank Optimization** | Replace heuristic weights with Nelder-Mead optimization on the paired validation OOF matrix. | Drives compartmental balance, lifting underperforming pathologies ($\Delta \approx +0.004$). |
| **Phase 4** | **Sub-400s Efficiency Lock** | Profile and optimize test-time DICOM decoding to maintain $< 400\text{s}$ execution. | Secures position for the **$18,000 Efficiency Prize** while contending for Leaderboard Gold. |
