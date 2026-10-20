# What Could Go Wrong & Prevent Reaching Top 10 / #1 — The Master Prevention Playbook

**Competition:** RSNA Knee Abnormality Detection Challenge (2026)  
**Target:** Top 10 / Gold Tier ($\ge 0.967$) & World #1 ($0.969$)  
**Current Baseline:** 0.958 (Rank 710 / 6,264; 1,571 teams tied $\ge 0.958$)  
**Safety Anchor:** Slot 1 (`57253840`, 0.958 PB) — Permanently Locked  

---

## Part I: Complete Root-Cause Analysis — Why Teams Get Trapped at 0.958

### 1. The Collinearity Trap (Correlation $\rho > 0.96$)
- **The Failure Mode:** Almost every public notebook on Kaggle uses the same two base backbones: CoAtNet-384 and ConvNeXt-Small (R7/R6). When ensembling models with identical training data and similar feature representations, their error residuals are collinear ($\rho > 0.96$).
- **Why It Prevents Top 10:** Mathematical ensembling theory states that ensemble variance reduction $\sigma^2_{\text{ens}} = \frac{1}{M}\sigma^2 + \frac{M-1}{M}\rho\sigma^2$. As $\rho \to 1.0$, ensemble gain drops to exactly zero. No matter how many weight variations are tested, the score remains 0.958.
- **The Solution:** True orthogonal inductive biases:
  - **ConvNeXt-Large-384** (198M params): Pure depthwise convolutions with 7x7 receptive fields. Excels at bone cortex, subchondral sclerosis, and osteophytes.
  - **Swin-Base-384** (88M params): Shifted-window multi-head self-attention. Excels at tracking non-local ligament paths (ACL/MCL) across 3D slice sequences.

### 2. The Spatial Resolution Nyquist Limit (224px vs 384px)
- **The Failure Mode:** Fast pipelines downsample knee MRI series to $224 \times 224$ pixels to save compute.
- **Physical Reality:** At a standard $140\text{ mm}$ Field of View (FOV):
  - At $224\text{ px}$: $1\text{ pixel} = 0.625\text{ mm}$. A subtle $0.8\text{ mm}$ meniscal radial tear spans only $1.28\text{ pixels}$. After bilinear interpolation and pooling, the lesion signature is obliterated into background noise.
  - At $384\text{ px}$: $1\text{ pixel} = 0.364\text{ mm}$. The same lesion spans $4.4\text{ pixels}$, allowing attention heads to localize the high-signal fluid intrusion.
- **The Solution:** Non-negotiable $384\times 384$ minimum resolution for all primary feature extractors.

### 3. Overfitting to the Gold-58 Public Anchor (Sample Size Fallacy)
- **The Failure Mode:** Tuning model hyper-parameters or ensembling weights based purely on the 58 official radiologist-annotated studies.
- **The Mathematical Reality:** In a cohort of $N=58$ studies with only 7 fractures and 9 MCL tears:
  - Flipping a single classification from true negative to false positive alters the class AUC by $\Delta \text{AUC} \approx \frac{1}{7 \times 51} \approx 0.0028$.
  - Tuning on Gold-58 fits stochastic noise and clinical idiosyncrasies rather than generalized pathology signatures.
- **The Solution:** Gold-58 is strictly a sanity check ($\ge 0.85$ Macro AUC). Model selection must be governed by 5-Fold Stratified Out-of-Fold (OOF) cross-validation across all 4,407 studies.

### 4. Calibration Mismatch & Raw Sigmoid Averaging
- **The Failure Mode:** Blending predictions by averaging raw sigmoid probabilities ($P = \frac{1}{2}(P_A + P_B)$) or applying power sharpening ($P^\alpha, \alpha > 1.0$).
- **The Disaster (Verified in V42):** V42 used raw probability ensembling and dropped from 0.958 to 0.943. Models trained with different loss functions (BCE vs Focal Loss) output completely different logit scales. The uncalibrated model with high-entropy outputs dominates the average, corrupting well-calibrated predictions.
- **The Solution (AGENTS.md Rule 2):** **Strict Percentile Rank Standardization** (`pd.Series(v).rank(method='average', pct=True)`). Rank transformation guarantees identical uniform distributions $[0, 1]$ regardless of model calibration temperature.

### 5. Private Test Data Distribution Shift & Scanner Variance
- **The Failure Mode:** The private test set contains MRI acquisitions from scanner manufacturers, field strengths (1.5T vs 3.0T), and sequence protocols unseen in the training distribution.
- **The Solution:**
  - Robust intensity percentiling ($p_1, p_{99}$) rather than absolute min-max normalization.
  - Multi-planar slice aggregation (Sagittal fluid-sensitive + Sagittal non-fluid + Coronal fluid + Coronal non-fluid + Axial).
  - Test-time flip augmentation (Flip-TTA) to eliminate lateral bias.

---

## Part II: 25 Fatal Failure Modes & How They Are Mitigated

| # | Fatal Failure Mode | Root Cause | Built-in Mitigation in Codebase |
| :--- | :--- | :--- | :--- |
| 1 | **Kaggle Kernel OOM** | Loading uncompressed full-volume 3D tensors into GPU RAM | Chunked slice windowing ($k=16$, `chunk_size=4`) + fp16 mixed precision |
| 2 | **Submission Timeout (>9h)** | Unbuffered DICOM reading or single-threaded disk I/O | Memory-mapped `.npy` / `.parquet` pre-extracted feature caches |
| 3 | **Interactive Test Mock Crash** | Hardcoded assertion `assert len(test_df) > 500` | Automated AST scan with `tools/validate_kernel_guardrail.py` |
| 4 | **Missing Auxiliary Dataset** | Kernel dataset mount path changes or dataset deleted | Fallback to anchor predictions if auxiliary weight path does not exist |
| 5 | **Extreme Prediction NaN/Inf** | Numerical underflow in logit sigmoid or BCE loss | Automatic clipping to $[0.0001, 0.9999]$ before CSV serialization |
| 6 | **Label Column Ordering Bug** | Alphabetical sort mismatch vs `sample_submission.csv` | Reindexing columns strictly against `sample_submission.csv` template |
| 7 | **Study ID Key Misalignment** | String vs Integer type casting mismatch in `StudyInstanceUID` | Enforced `dtype={'StudyInstanceUID': str}` across all pandas operations |
| 8 | **Single-Class Zero Variance** | Model outputs uniform constant for rare class (e.g. MCL) | `pct_rank` fallback to $0.5$ if `np.all(vals == vals[0])` |
| 9 | **Loss of Slot 1 Safety Anchor** | Agent replaces 0.958 submission with unverified model | Hard Rule 1 of `AGENTS.md`: Never deselect Slot 1 unless candidate $\ge 0.9610$ |
| 10 | **Gradient Exploding in Large Backbones** | Unscaled gradients in 198M-parameter ConvNeXt-Large | Gradient norm clipping (`torch.nn.utils.clip_grad_norm_(1.0)`) |
| 11 | **Catastrophic Forgetting** | Single high learning rate across pre-trained backbone | Differential learning rates ($2\times 10^{-5}$ backbone, $2\times 10^{-4}$ head) |
| 12 | **Colab Disconnect During Training** | Idle browser tab or transient WebSocket disconnect | Automated off-box checkpoint pushing via Kaggle CLI at each epoch |
| 13 | **Disk Quota Exceeded on Colab** | Uncompressed temporary DICOM extractions | Streaming dataset reading with immediate intermediate file cleanup |
| 14 | **GPU Kernel Crash on Test Multi-GPU** | `torch.cuda.set_device` hardcoded to device 0 | Dynamic `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |
| 15 | **Corrupt Weights Upload** | Partial network write during Kaggle dataset versioning | SHA-256 hash verification and PyTorch checkpoint load smoke tests |
| 16 | **Sub-500s Efficiency Trap** | Constraining main leaderboard models to satisfy efficiency rules | Decoupled tracks: V45 defends $18k efficiency; V48 targets $59k main prize |
| 17 | **Clinical Prior Overfitting** | Hardcoding biomechanical boosts ($P(\text{ACL}) \times P(\text{Contusion})$) | Excision of clinical heuristic multipliers (ADR-002: dropped V12 to 0.953) |
| 18 | **Power Sharpening Degradation** | Applying $P^\alpha (\alpha > 1)$ to sharpen confident predictions | Prohibited power sharpening (V42 dropped to 0.943; strictly linear rank) |
| 19 | **Over-Reliance on Weak Pseudo-Labels** | Training on unweighted raw LLM radiology report extractions | Super-consensus soft labels with confidence weight masking |
| 20 | **Missing Slices in Test Studies** | Incomplete series acquisitions in clinical practice | Dynamic zero-padding with missing-slot indicator mask |
| 21 | **Timm Version Incompatibility** | Model architecture name deprecated across library versions | Pinned architecture strings and local custom MIL head definitions |
| 22 | **Daily Quota Exhaustion** | Submitting unverified or broken notebooks, wasting 5/5 quota | Mandatory AST and mock testing via `validate_kernel_guardrail.py` |
| 23 | **Timezone Miscalculation** | Missing competition deadline due to UTC vs IST confusion | Fixed countdown in `docs/LIVE_STATE.md`: Deadline Oct 22 23:59 UTC |
| 24 | **Secret / Token Leakage** | Committing API tokens or `.env` files into public repos | Enforced `.gitignore` masking of all credentials, `.pt`, and `.parquet` |
| 25 | **Silent Script Failure in Submissions** | Unhandled exception in auxiliary model forward pass | Fail-soft wrapper: auxiliary failures caught gracefully, anchor used |

---

## Part III: 10 Permanent Assets for the Long Run

To ensure continuous, compound engineering progress across AI agents and sessions:

1. **`AGENTS.md` (Autonomous Constitution):**  
   Mandatory operating protocol read first by every AI agent. Enforces logging, verification, and non-negotiable safety rules.
2. **`docs/LIVE_STATE.md` (Telemetry Ground Truth):**  
   Auto-generated via `tools/state_snapshot.py` directly from Kaggle API. The single source of truth for ranks, quota, and scores.
3. **`docs/LIVE_STEP_LOG.md` (Append-Only Audit Trail):**  
   Every meaningful step logged with observed evidence, command outputs, and the single next action.
4. **`tools/validate_kernel_guardrail.py` (Pre-Flight AST Scanner):**  
   Validates notebooks against fatal asserts, test size assumptions, NaN outputs, and mock test datasets before submission.
5. **`tools/monitor_all_pending_subs.py` (Persistent Leaderboard Daemon):**  
   Background daemon tracking scoring workers up to 6 hours, recording official public scores automatically upon completion.
6. **`tools/state_snapshot.py` (API Telemetry Ingest):**  
   Queries Kaggle API for exact competition standing, cutoff thresholds, and remaining daily submissions.
7. **`cyruspinto/rsna-knee-engine12-convnext-large-weights` (Kaggle Dataset):**  
   Houses the 198M-parameter ConvNeXt-Large-384 checkpoint ($0.8808$ Gold AUC) with depthwise convolutional inductive bias.
8. **`cyruspinto/rsna-knee-engine13-swin-base-weights` (Kaggle Dataset):**  
   Houses the 88M-parameter Swin-Base-384 checkpoint ($0.8577$ Gold AUC) with shifted-window self-attention inductive bias.
9. **`colab_engine12_fold1_master.ipynb` (Cross-Validation Suite):**  
   Complete 5-Fold Stratified cross-validation pipeline with automated off-box weight publishing.
10. **`submissions/v48_apex_quad_arm_ensemble` (Production Sovereign Ensemble):**  
    Fuses ConvNeXt-Large-384, Swin-Base-384, Raptor 384/224, and ConvNeXt R6 using pure percentile rank standardization.
