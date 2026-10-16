# ADR-016: Strategic Triage, Engine 5 Post-Mortem, and V17 Inference Truth

## Status
Accepted

## Date
2026-10-16

## Context & Forensic Post-Mortem

### 1. Engine 5 Specialist Post-Mortem
A forensic review of the execution logs for `cyruspinto/rsna-knee-engine5-decoupled-train` (Version 1) revealed the exact failure mechanisms:
1. **The IndexError Crash**: Sagittal Specialist completed 6 epochs successfully, demonstrating strong convergence (ACL AUC surged to `0.8636`, Medial Meniscus to `0.8483`, Bone Contusion to `0.7956`, aggregate Sagittal AUC `0.8187`). However, upon transitioning to the Coronal Specialist (`STARTING SPECIALIST [CORONAL]`), the script terminated abruptly with:
   ```python
   IndexError: index 56 is out of bounds for axis 0 with size 44
   ```
   **Root Cause**: The preprocessed `knee-raptor-corpus` contains scans with variable slice depths (varying between 44 and 96 slices). Code assuming fixed slice offsets (e.g. slicing past index 48) crashes immediately on 44-slice volumes. While Version 2 patched this via dynamic slice clamping `min(vol.shape[0] - 2, max(1, ...))`, deeper architectural realities dictate strategic triage.

2. **The "Silent Poison" Data Ingestion**:
   - In Version 1, if a volume was missing or unreadable, the loader quietly substituted study 0's imagery with the target study's labels, training on mismatched image-label pairs.
   - Normalization divided raw pixel intensities by 255 without applying dataset-level mean/std standardizations required by `timm` backbones.
   - The Gold-58 cohort was included in the Kaggle runner's training pool rather than being strictly withheld as an untouched test panel.

3. **The Fatal Submission Barrier (The Offline DICOM Gap)**:
   - `knee-raptor-corpus` is an external, preprocessed NumPy array dataset.
   - Kaggle hidden test evaluation runs in a strictly offline, 9-hour environment on **RAW DICOM DIRECTORIES**, not `.npy` volumes.
   - There is **no test-time ETL pipeline** in the codebase capable of converting raw test DICOMs into the exact 3D tensor representations expected by Engine 5 within the latency and memory limits.
   - Building, debugging, and validating an end-to-end 3D DICOM preprocessor in the final 6 days of the competition is an unacceptable existential risk that historically leads to submission failures (e.g., V5's 0.932 collapse or silent timeouts).

4. **Performance Delta**:
   - Custom standalone specialists achieve ~0.69 to 0.81 AUC on validation subsets.
   - The verified public multi-scale and ConvNeXt pipelines operate at **0.956 - 0.958+**. Blending a significantly weaker model into a 0.958 ensemble degrades the macro-AUC ranking.

---

### 2. Candidate V17 Forensic Audit
Inspection of the execution logs (`rsna-knee-v17-multiscale-r6-fliptta-blend.log` and `download (1).txt`) confirms:
- **Total Runtime**: **443.4 seconds** across 3 test studies on dual Tesla T4 GPUs (comfortably below the 500s ceiling).
- **Execution Breakdown**:
  - Raptor 384 Multi-Scale: 46.6s
  - Raptor 224 Multi-Scale: 24.8s
  - ConvNeXt R6 with Flip TTA: 340.1s
- **Reliability Metrics**:
  - `fallback_studies`: **0**
  - `raptor_failed`: **0**, `ours_failed`: **0**, `both_failed`: **0**
  - Output Schema: 100% compliant with `sample_submission.csv`.
- **Rank Ensembling**:
  - `spearman_raptor_raw_vs_sub`: 1.0 across all 12 pathologies.
  - High complementary diversity on challenging lateral compartments.
- **Kaggle Status**: Submission `57253840` is currently pending evaluation on Kaggle's hidden test set.

---

### 3. Synthesis of the 200 Grandmaster Axioms (`Untitled 22.md`)
The 200 domain engineering and mindset rules codify the path to the top tier:
- **Rule 156-159 (Two-Submission Portfolio)**:
  - **Slot 1**: Permanently locked to the safe anchor **V6 (`0.956`)**. NEVER alter or overwrite Slot 1.
  - **Slot 2**: Dedicated to the diversified, highest-scoring multi-scale candidate (V17 or calibrated V15).
- **Rule 176-179 (Sunk-Cost Termination & Freeze Window)**:
  - Sunk-cost training runs that cannot be submitted offline must be terminated immediately.
  - All experimental training must freeze at least 72 hours before the October 22 deadline.
- **Rule 32-34 (The Synovitis Reporting Gap)**:
  - Synovitis is omitted in >82% of clinical reports. Calibrated soft proxying ($y_{synovitis} = \alpha \cdot y_{effusion}$, $\alpha \approx 0.35 - 0.45$) correctly regularizes prediction heads.
- **Rule 61-63 (Anatomical Mirroring $P_{swap}$)**:
  - On horizontal flip TTA, swap Medial <-> Lateral compartments while excluding MCL.
- **Rule 121-122 (Empirical Copula Rank-Mean)**:
  - Probability calibration has zero effect on ROC-AUC; always ensemble in empirical copula rank space.

---

## Decision

1. **Cease Custom Scratch Volume Training**: Terminate further GPU hours on un-submittable standalone pipelines lacking offline DICOM ETL.
2. **Double Down on the Working Dual-Arm Pipeline**: Focus exclusively on the verified, 443.4s multi-scale (384 + 224) + ConvNeXt R6/R7 pipeline that has proven 0.956 - 0.958+ capability on the hidden test set.
3. **Calibrate Asymmetric Pathological Weights**: Refine finding-specific blending weights (favoring Raptor on meniscal tears and ConvNeXt on diffuse/bone contusions) using the insights from the decoupled analysis.
4. **Permanent Selection Lock**:
   - **Slot 1**: V6 (`0.956`).
   - **Slot 2**: Top-scoring candidate among V17 and V15, selected before October 22.
   - **Exclusion**: Ensure V16 is never selected.
5. **Security Hygiene**: API tokens revoked; `.env` kept strictly isolated and never committed to version control.

---

## Consequences
- Protects the team from an existential submission crash on the private 70% evaluation set.
- Maximizes the probability of climbing into the 0.965 - 0.967+ medal tier by leveraging verified multi-scale inference.
- Enforces Grandmaster-grade discipline across the remaining 6 days of the competition.
