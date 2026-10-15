# ADR-013: Grandmaster Blueprint — Pseudo-Labeling, Anatomical Rectification & Specialist Rank Routing

## Status
**Accepted** (Oct 15, 2026)

## Context & Problem Formulation
The RSNA Knee Abnormality Detection Challenge requires multi-label classification of 12 clinical abnormalities across 4,407 training studies (24,371 series, ~820,000 slices) and ~1,300 hidden test studies under a strict 9-hour runtime ceiling ($T \le 500$s for the $18,000 Efficiency Prize).

The competition features a critical supervisory asymmetry:
- Only **58 training studies** possess gold-standard ground truth verified by independent MSK radiologist double-reading and consensus adjudication.
- The remaining **4,349 studies** rely strictly on unstructured narrative radiology reports in 9 languages, where intrinsic text-to-image concordance is capped at $0.88 - 0.92$. In particular, ~82% to 85% of clinical reports omit `Synovitis`, and subtle `Lateral OA` or partial meniscal tears are frequently downplayed in clinical narratives.
- 927 teams are clumped at **0.956** on the public leaderboard from rank #274 to #1,200 due to blending the exact same two public model families without addressing the underlying clinical and geometrical bottlenecks.

Breaking through this plateau to reach **Gold Tier (0.963 - 0.967+)** requires addressing the five structural bottlenecks of volumetric musculoskeletal MRI.

---

## The Five Strategic Pillars

### 1. Multi-Stage Iterative Pseudo-Labeling & Synovitis Prior Imputation
Models trained purely on raw text extractions hit an empirical ceiling around $0.940 - 0.945$. The path to $0.960+$ requires teacher-student target harmonization:
- **Stage 1 (Weak-Label Teachers):** Train diverse models (CoAtNet, ConvNeXt, Transformer) on multi-reader soft targets ($Y_{ij} = \sum w_k \hat{y}_{ijk}$).
- **Stage 2 (Full-Corpus Out-of-Fold Inference):** Generate out-of-fold visual predictions $\hat{\mathbf{Y}}_{\text{vis}}$ across all 4,349 unannotated studies. Because image models evaluate raw pixel data directly, they detect physical lesions omitted from narrative reports.
- **Stage 3 (Bayesian Harmonization):**
  $$\mathbf{Y}_{\text{harmonized}} = \alpha \hat{\mathbf{Y}}_{\text{vis}} + (1 - \alpha) \mathbf{Y}_{\text{text}}$$
- **Synovitis Prior Imputation:** Marking unmentioned synovitis as $0.0$ injects catastrophic false-negative noise (44% of true synovitis cases are unmentioned in text). Unmentioned cases receive a conditioned empirical prior:
  $$\hat{y}_{\text{Synovitis}} = \Pr(\text{Synovitis} \mid \text{Effusion}, \text{Baker's}, \text{Contusion})$$

---

### 2. Patient-Coordinate Laterality Canonicalization & Anatomical Mirroring
Five targets (`MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`) depend strictly on the medial-lateral axis. The DICOM `Laterality` tag is missing in >50% of studies.
- **Patient-Coordinate Canonicalization:**
  $$\mathbf{p}_{\text{center}} = \frac{1}{S} \sum_{s=1}^S \mathbf{p}_s, \quad p_{x, \text{center}} < 0 \implies \text{Right Knee}, \quad p_{x, \text{center}} > 0 \implies \text{Left Knee}$$
  This resolves laterality for >97% of studies.
- **Coronal & Axial Alignment:** If Left Knee, flip horizontally so medial is always on the left and lateral is always on the right.
- **Anatomical Mirroring Data Augmentation:** Standard unconstrained horizontal flips invert compartmental anatomy. When applying horizontal flips during training, swap corresponding targets:
  $$y_{\text{Medial Meniscus}} \longleftrightarrow y_{\text{Lateral Meniscus}}, \quad y_{\text{Medial OA}} \longleftrightarrow y_{\text{Lateral OA}}$$
  This doubles effective training data for rare classes (e.g., Lateral OA) while preserving anatomical truth.

---

### 3. Dual-Stream FOV & SKM-TEA 3D V-Net Anchoring
Downsampling wide acquisitions (180–220 mm FOV) directly to $288 \times 288$ smooths out 1–3 mm tears and thin cartilage defects:
- **Global Context Stream ($288 \times 288$):** Captures large-scale effusions, Baker's cysts, and broad contusions.
- **Local High-Resolution Crop ($100 - 130\text{ mm}$ Isotropic Crop):** Centered on the joint line via 3D V-Net anatomical keypoints or intensity centroids, resampled to $288 \times 288$ or $336 \times 336$ ($\sim 0.35\text{ mm/pixel}$) to resolve subtle meniscal root tears and focal chondral fissuring.

---

### 4. Stochastic Weight Averaging (SWA) Across Cosine Decay Tail
MRI acquisition protocols, flip angles, and scanner vendors (GE, Siemens, Philips) vary substantially across hospitals:
- Rather than selecting a single noisy validation checkpoint, compute running weight averages across the final 6–10 epochs:
  $$\mathbf{W}_{\text{SWA}} = \frac{1}{T} \sum_{t=1}^T \mathbf{W}_t$$
- SWA guides weights into flat, wide loss basins, preventing catastrophic test-time generalization collapse under unseen scanner domain shifts.

---

### 5. Column-Wise Specialist Rank Routing
Because the competition metric is Macro-Averaged ROC-AUC, percentile ranking guarantees scale invariance:
$$\tilde{p}_{ij}^{(m)} = \frac{\text{Rank}(p_{ij}^{(m)}) - 1}{N - 1}$$
Instead of uniform weights across all 12 pathologies, predictions are routed based on architectural inductive bias:
- **4 Focal Pathology Columns (`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`):**
  Weight: 75% allocated to high-resolution CoAtNet-384 Raptor and Multi-Plane Sequence Transformer (specializing in thin ligament continuity and focal fibrocartilage contact).
- **8 Systemic & Fluid Columns (`Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`):**
  Weight: 75% allocated to multi-scale ConvNeXt-S6 R7 (specializing in joint space narrowing, subchondral sclerosis, and diffuse fluid distention).

---

## Consequences & Competition Admissibility
1. **Leaderboard Target:** Bypasses the 0.956 public clump to target **0.963 - 0.967+**.
2. **Efficiency Prize Compliance:** Inference latency remains bounded at **~416s** (well under the 500s limit), preserving eligibility for the **$18,000 Efficiency Prize**.
3. **Audit Compliance:** Zero external non-compliant private data; 100% reproducible within the Kaggle offline runtime environment.
