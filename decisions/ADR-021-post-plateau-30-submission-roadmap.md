# ADR-021: 6-Day 30-Submission Top-1 Strategic Roadmap (Oct 17 – Oct 22)

## Status
ACCEPTED / STRATEGIC BASELINE

## Context
1. Competition Deadline: **October 22, 2026, 23:59:00 UTC**.
2. Total Submissions Remaining: 1 today (Oct 16) + 30 across 6 days (5/day) = **31 total submissions**.
3. Current Incumbent:
   - Slot 1: Candidate V6 (`0.956` Public LB) [PERMANENT BASELINE ANCHOR]
   - Slot 2: Candidate V17 (`0.958` Public LB) [CHALLENGER TARGET: $\ge 0.960$]
4. Breakthrough Thesis:
   - Ensembling downsampled $224 \times 224$ models is bounded by spatial Nyquist limits on sub-millimeter knee structures.
   - Genuine podium jumps ($0.958 \to 0.965+$) require:
     1. High-resolution joint line crops ($512 \times 512$, $0.35\text{ mm/pixel}$).
     2. 3D inter-slice sequence continuity (ConvNeXt-S + BiLSTM).
     3. Biomechanical relational cross-attention (O'Donoghue's Triad conditioning).
     4. Anatomical laterality canonicalization & horizontal flip target swapping.

---

## 6-Day Execution Cadence (5 Submissions / Day)

### Day 1: October 17 — Quint-Engine Calibration & Laterality Mirroring
- **Sub 1:** Candidate V22 Baseline (Quint-Engine Master Blend).
- **Sub 2:** Candidate V22 + Laterality Invariant Canonicalization (Medial compartment left-aligned).
- **Sub 3:** Candidate V22 + Horizontal Mirror Augmentation at test-time ($y_{\text{medial}} \leftrightarrow y_{\text{lateral}}$).
- **Sub 4:** Temperature-Scaled Focal Probability Calibration (optimizing ECE on double-read gold studies).
- **Sub 5:** OOF Stacked Calibrated Blend across all 5 engines.

### Day 2: October 18 — Biomechanical Relational Cross-Attention
- **Sub 6:** Engine 6 Sagittal Specialist with Biomechanical Relational Head.
- **Sub 7:** Engine 6 Coronal Specialist with Compartmental OA Anti-Symmetric Head.
- **Sub 8:** Dual-Head Coupled Quint-Engine Blend.
- **Sub 9:** Soft-Label Target Harmonized Model (incorporating visual teacher OOFs).
- **Sub 10:** Daily Best Ensemble Checkpoint.

### Day 3: October 19 — Multi-Scale Triplet Ensembling (512 + 384 + 224)
- **Sub 11:** 512 Native Joint Crop Model standalone benchmark.
- **Sub 12:** Multi-Scale Concatenation: 512 Joint + 384 Whole Volume.
- **Sub 13:** Multi-Scale Concatenation: 512 Joint + 224 Global Context.
- **Sub 14:** Tri-Scale Pyramidal Ensemble (512 / 384 / 224).
- **Sub 15:** Daily Best Ensemble Checkpoint.

### Day 4: October 20 — 5-Fold Engine 6 Checkpoint Ensemble
- **Sub 16:** Fold 1 & 2 Engine 6 Checkpoint Averaging.
- **Sub 17:** Fold 3 & 4 Engine 6 Checkpoint Averaging.
- **Sub 18:** Full 5-Fold SWA Engine 6 Master Weights.
- **Sub 19:** 5-Fold Quint-Engine Super Blend.
- **Sub 20:** Daily Best Ensemble Checkpoint.

### Day 5: October 21 — Efficiency Prize & Hard Constraint Stress Testing
- **Sub 21:** Quantized FP16 / TensorRT Engine Evaluation ($< 350\text{s}$ hidden test latency).
- **Sub 22:** Memory-optimized single T4 submission (guaranteeing zero OOM).
- **Sub 23:** Extreme outlier robustification (handling missing DICOM tags).
- **Sub 24:** Pre-Final Candidate A (Conservative high-certainty blend).
- **Sub 25:** Pre-Final Candidate B (Aggressive high-diversity blend).

### Day 6: October 22 — Final Selection & Freeze (Anti-Queue Trap Protocol)
- **Submission Window:** **00:00 to 12:00 UTC ONLY** (Never submit in the final 6 hours).
- **Sub 26:** Final Verification Run A.
- **Sub 27:** Final Verification Run B.
- **Sub 28:** Tie-breaker / Sensitivity Check.
- **Sub 29 & 30:** Reserved as buffer for unforeseen Kaggle platform anomalies.
- **Final Selection Locked by 12:00 UTC (12 hours before close):**
  - **Slot 1:** Most robust verified model (Candidate V6 or proven successor).
  - **Slot 2:** Highest scoring empirical public LB candidate ($\ge 0.963$).
