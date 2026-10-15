# SDD Plan: RSNA Knee Abnormality Detection Breakthrough (0.956 -> 0.963+)

## Global Constraints
1. **Submission Time Limit:** Complete test inference on 1,300 studies must execute in $T \le 450$ seconds on Kaggle Dual T4 / P100 GPUs (Safety margin for $18,000 Efficiency Prize $T \le 500$s limit).
2. **Deterministic Evaluation:** Never evaluate models or tune hyperparameters solely on the 58 Gold studies (high variance noise). Use 5-fold Stratified Group-KFold on the 4,349 studies.
3. **No Overfitting Priors:** Exclude static additive heuristic priors ($\pm 0.001$). All post-processing must be learned through temperature scaling or power-rank fusion.
4. **Data Isolation:** No hardcoded secrets, no raw DICOM git commits. Only clean code and distilled weights.

---

## Tasks

### Task 1: 5-Fold Stratified Group-KFold Cross-Validation Framework
- **Goal:** Create a rock-solid offline validation harness on the 4,349 distilled training studies grouped by `StudyInstanceUID` to replace the 58-study noisy feedback loop.
- **Deliverables:** `validation/cv_framework.py` with multi-label StratifiedGroupKFold, out-of-fold AUC computation per condition, and macro-AUC.
- **Verification:** Unit test with synthetic multi-label distribution verifying zero study leakage across folds and reproducible AUC scores.

### Task 2: Multi-Plane Feature Alignment Engine (Engine 3 Slice Transformer)
- **Goal:** Bridge the 44-slice train vs 96-slice multi-plane test inference gap (ADR-004) by adding multi-sequence position embeddings (Sagittal, Coronal, Axial) to the Attention-MIL backbone.
- **Deliverables:** `training/engine3_slice_transformer/model_multiplane.py` supporting variable slice counts per sequence with cross-plane attention pooling.
- **Verification:** Forward pass test with both 44-slice and 96-slice multi-channel tensors verifying exact shape outputs $(B, 12)$ and valid gradient flow.

### Task 3: Dual-Engine + Engine 3 Tri-Hybrid Inference Pipeline Benchmark
- **Goal:** Integrate Engine 3 with Raptor CoAtNet and Heliosli ConvNeXt R7 into a hardened inference script that runs under 420 seconds.
- **Deliverables:** `submissions/v13_tri_hybrid_aligned/submission.py` with FP16 inference, dynamic batching, and uniform rank blending.
- **Verification:** Offline mock test benchmarking 100 dummy studies verifying execution time $\le 30$ seconds and valid submission format.
