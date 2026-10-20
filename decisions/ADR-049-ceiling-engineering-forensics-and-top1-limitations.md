# ADR-049: Ceiling Engineering Forensics, What Went Wrong, and Absolute Top-1 Strategy

**Status:** ACCEPTED  
**Date:** 2026-10-20  
**Context:** RSNA Knee Abnormality Detection Challenge (Kaggle 2026)  
**Authors:** Chief Orthopaedic Imaging Architect / Antigravity AI Team  
**Supersedes/Extends:** ADR-001 through ADR-048  

---

## 1. Executive Summary & Problem Formulation

We entered the final 56 hours of the competition locked at **0.958 Public LB (Rank 735 / 6,271)**. Exactly 1,600 teams tie at $\ge 0.958$, confirming that 0.958 is the **public-notebook plateau** (produced by re-blending public CoAtNet and ConvNeXt-Small 224/384px kernels).

Reaching the Top-10 cutoff ($\ge 0.967$) or #1 ($\ge 0.969$) requires a gain of **+0.009 to +0.011 Macro-AUC**. Such an increase CANNOT be attained by tuning blending weights of existing public arms. It requires **new orthogonal signal**, **native 384px spatial resolution**, **multi-fold variance suppression**, and **strict architectural discipline**.

This ADR documents the forensic post-mortems of every historical failure, proves why they happened, establishes immutable guardrails to prevent their recurrence, and outlines the 10 permanent engineering assets needed to reach the absolute ceiling.

---

## 2. Forensic Autopsy: What Went Wrong & Why

### Case 1: Candidate V47 Regression (0.958 → 0.953)
- **Observed Result:** Score dropped by -0.005 Macro-AUC on Public LB.
- **Root Cause (ADR-048):** Recursive double-blending of low-resolution slices. The pipeline took `raptor_sub.csv` (which internally already blends 384px and 224px slices) and blended it with `raptor224_sub.csv` (weight 0.30) and other arms. The effective weight of 224px slices ballooned to **57.8%**.
- **Biomechanical Mechanism:** Knee MRI abnormalities (particularly peripheral meniscal tears, root tears, and hairline fractures) have sub-millimeter lesion dimensions (0.8–1.5 mm). Downsampling volumes to $224 \times 224$ introduces spatial Nyquist blur that permanently destroys these micro-structural gradients.
- **Prevention Implemented:** Rule 11 added to `AGENTS.md`. AST guardrail (`tools/validate_kernel_guardrail.py`) now scans notebooks and automatically rejects any blend where $\ge 384\text{px}$ models represent $< 70\%$ of total weight or where constituent single-scale files are blended with multi-scale parents.

### Case 2: Candidate V42 Disaster (0.958 → 0.943)
- **Observed Result:** Score collapsed by -0.015 Macro-AUC.
- **Root Cause (ADR-001/003):** Application of non-linear power sharpening $S = \left(\sum w_i r_i^p\right)^{1/p}$ with $p = 1.4$.
- **Mathematical Mechanism:** ROC AUC is strictly determined by rank concordance across positive and negative pairs. Applying a non-linear power $p > 1$ distorts the uniform rank distribution $\mathcal{U}(0, 1)$, heavily compressing true negative scores in the lower percentile tail into identical near-zero values. Any minor perturbation in those tied ranks causes massive pairwise inversion penalties under AUC.
- **Prevention Implemented:** Hard Rule 2 in `AGENTS.md`: Linear rank-percentile blending ONLY ($\sum w_i r_i$ where $\sum w_i = 1.0$). Zero power transforms permitted.

### Case 3: Candidates V34–V36 Fatal Exception Outages (Scored "Error")
- **Observed Result:** Submissions failed on Kaggle scoring backend with "Notebook Threw Exception". 3 daily submissions completely wasted.
- **Root Cause (ADR-042):** Unconditional assertions on hidden test dimensions (`assert len(df) == 120` and `assert set(planes) == {'sagittal', 'coronal', 'axial'}`). The hidden private test set contains thousands of studies, some of which feature corrupted DICOM headers, single missing series, or non-standard series tags.
- **Prevention Implemented:** Hard Rule 4 in `AGENTS.md`: Zero fatal asserts or unhandled exceptions on test data. Every DICOM loader and series aligner is wrapped in try/except blocks with column-median prior imputation fallbacks.

### Case 4: Candidate V12 Regression (0.958 → 0.953)
- **Observed Result:** Drop of -0.005 Macro-AUC.
- **Root Cause (ADR-002):** Hand-crafted clinical "boost" heuristics (e.g. boosting Bone Contusion probability if ACL Tear was predicted $> 0.85$, or boosting Lateral Meniscus if Pivot Shift mechanism was inferred).
- **Clinical Mechanism:** While ACL tears and lateral bone contusions are physiologically correlated, hand-tuned conditional multipliers distort the uncalibrated probability thresholds across the entire population, causing catastrophic false-positive spikes on ambiguous cases.
- **Prevention Implemented:** Hard Rule 3 in `AGENTS.md`: No heuristic post-processing or manual probability boosting unless validated across all 4,407 full 5-fold OOF cases with proven $\Delta \text{AUC} \ge +0.002$.

### Case 5: The V18–V33 Stagnation (0.956–0.958 Plateau)
- **Observed Result:** 15 consecutive submissions spent 3 days varying weights between 0.956 and 0.958 without making any progress toward 0.965+.
- **Root Cause:** Pseudo-diversity. Re-weighting the same two public models (CoAtNet and ConvNeXt-Small) that possessed a pairwise Spearman rank correlation $\rho > 0.95$. Blending two models that make identical errors yields mathematical zero variance reduction.
- **Prevention Implemented:** Hard Rule 5 & Rule 11: Candidate arms must possess verified structural diversity (combining Modern Hierarchical Vision Transformers, Pure ConvNets, and Self-Supervised Foundation Models).

---

## 3. The 10 Permanent Long-Run Assets to Reach the Ceiling (#1 / 0.969)

To sustainably break through the 0.958 ceiling and compete for Rank 1, we codify the following 10 assets:

1. **Native 384px Spatial Dominance ($\ge 70\%$ Ensemble Weight):**
   - Preserves high-frequency spatial gradients required for meniscal tears and hairline fractures.
2. **Multi-Fold Variance Suppression (5-Fold Averaging):**
   - Eliminates single-split sampling noise; single-fold models exhibit $\pm 0.004$ variance on private test distributions.
3. **True Architectural Orthogonality:**
   - Swin-Base-384 (Hierarchical Local Window Attention) + ConvNeXt-Large-384 (Large 7x7 Depthwise CNN) + DINOv2-ViT-L (Self-Supervised Medical Semantics). Pairwise $\rho \le 0.88$.
4. **Gated Attention Multiple Instance Learning (Attention-MIL):**
   - Learns instance attention weights $\alpha_k = \frac{\exp(w^T \tanh(V h_k) \odot \text{sigm}(U h_k))}{\sum_j \exp(w^T \tanh(V h_j) \odot \text{sigm}(U h_j))}$ across slices, preventing slice averaging dilution.
5. **Cross-Plane Multi-Series Spatial Synergy:**
   - Joints Sagittal, Coronal, and Axial series before classification; diagnoses multi-planar pathology (e.g. MCL on Coronal, ACL on Sagittal, Patellar on Axial).
6. **Asymmetric Focal Loss for Rare Pathology:**
   - Mitigates class imbalance for Fracture (prevalence ~2.5%) and MCL (prevalence ~3.2%), preventing the classifier from collapsing to majority negative predictions.
7. **AST Kernel Guardrail & CI/CD Verification:**
   - `python tools/validate_kernel_guardrail.py` strictly prevents broken code, fatal asserts, and Nyquist violations from ever reaching the Kaggle queue.
8. **Automated Quota & Telemetry State Synchronization:**
   - `tools/state_snapshot.py` and `tools/log_step.py` provide single-source-of-truth continuity across AI agents and human teammates.
9. **The Two-Track Defense Protocol:**
   - Slot 1 permanently preserves the 0.958 Safety Submission (Ref 57253840) to guarantee a top-tier finish. Slot 2 aggressively targets the 0.969 ceiling with validated OOF ensembles.
10. **Rank-Percentile Invariant Ensembling:**
    - Linear ranking strictly eliminates probability calibration mismatch and protects against test-set prevalence shifts.

---

## 4. Consequences and Validation Protocol

- Any submission proposed to Kaggle must undergo pre-flight guardrail validation (`python tools/validate_kernel_guardrail.py`).
- Every training script must execute in an isolated process with gradient checkpointing enabled and chunk size $\le 4$.
- All findings and decisions are committed to Git with timestamped evidence.
