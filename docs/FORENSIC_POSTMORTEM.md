# RSNA Knee Abnormality Detection: Forensic Post-Mortem & Diagnostic Audit

Comprehensive forensic investigation and technical retrospective analyzing regressions, runtime anomalies, and software failures encountered during the engineering lifecycle.

![Evolutionary Trajectory](assets/figure1_evolution_trajectory.png)

## Executive Audit Summary

Engineering systems at the frontier of competitive machine learning inevitably encounter failure modes. Transparent post-mortem documentation is essential to institutionalizing engineering invariants.

| Ref | Incident Description | Score Impact | Root Cause | Permanent Invariant Fix |
| :---: | :--- | :---: | :--- | :--- |
| **#1** | **V50 Zero-Score Catastrophe** | 0.958 $\to$ 0.491 | Test IDs queried in training corpus | Live DICOM streaming only; AST ban on corpus test lookup |
| **#2** | **Triple Rank Quantization** | 0.958 $\to$ 0.957 | `rank_matrix()` called 3 times in pipeline | Continuous Probit fusion; zero intermediate ranking |
| **#3** | **Double Coupling Leakage** | False Positive Inflation | ACL coupling run in Stage 4 and Stage 6 | Orthogonal coupling matrix executed once at output |
| **#4** | **16-Window Feature Dilution** | 0.958 $\to$ 0.953 | Auxiliary non-OAI over-blended at 15% | Calibrated Versia_7 reader weights ($0.003$ to $0.035$) |
| **#5** | **4-vCPU GIL Thrashing** | Latency +109s ($551\text{s} \to 660\text{s}$) | 12 threads on 4 CPUs causing context thrash | Worker pool locked to 6; chunk size locked to 16 |
| **#6** | **Kaggle 20GB Mount Failure** | Container Init Crash | 12+ separate FP32 weight datasets | Weight-Space Model Soup in FP16 ($< 560\text{ MB}$ total) |

---

## Deep-Dive Diagnostic Case Studies

### Case Study 1: The V50 Zero-Score Catastrophe (LB 0.4910)
* **Symptom**: Candidate V50 committed cleanly in 279.8 seconds but scored an abysmal **0.4910 Macro ROC-AUC** on the public leaderboard.
* **Forensic Trace**:
  ```python
  # Flawed logic in V50:
  if uid in id2row and va is not None:
      row = id2row[uid]
      vol = va[row]
  else:
      # Fallback when UID is absent:
      preds[idx] = np.random.beta(0.5, 2.0, size=12)
  ```
  `all_vols.npy` contains strictly the 4,407 training studies. When the kernel executed on Kaggle against hidden test set studies, `uid in id2row` evaluated to `False` for **100% of the test cohort**. The script silently fell back to generating pseudo-random beta noise (`np.random.beta(0.5, 2.0)`), producing an AUC near 0.5000.
* **Root Cause**: The automated agent assumed `all_vols.npy` was a unified multi-dataset cache rather than an offline training-only artifact.
* **Permanent Invariant**: Hidden test studies must stream and decode directly from `test_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm`. The AST Pre-Flight Guardrail now explicitly checks that test paths resolve to live DICOM structures.

---

### Case Study 2: The Triple Rank-Quantization Cliff (0.9580 to 0.9570)
* **Symptom**: In Candidates V53 and V56, an upgraded ensemble incorporating the 2.5D ConvNeXt Reader dropped from 0.9580 to 0.9570.
* **Forensic Trace**:
  In `_v53_sovereign.py`:
  1. `r_anchor = rank_matrix(anchor_df.values)`
  2. `final_ranked = rank_matrix(final_mat)`
  3. `return rank_matrix(out)` inside `apply_biomechanical_coupling()`
* **Mechanism**:
  On small test batches ($N = 3$ to $120$ studies), calling `rank(pct=True)` three consecutive times forces smooth continuous values into coarse discrete fractions ($0.3333, 0.6667, 1.0000$). For example, if Model 1 and Model 2 agreed that a patient had high effusion probability but differed by only $0.0333$ ($0.8500$ vs $0.8167$), the sequential re-ranking snapped this minor margin into a massive $0.3333$ step cliff ($1.0000$ vs $0.6667$), destroying the subtle tie-breaking signals of auxiliary models.
* **Permanent Invariant**: Preserved continuous $[0, 1]$ margins throughout all intermediate blending steps; applied continuous Probit normal transforms ($\Phi^{-1}(p)$) rather than discrete step quantization.

---

### Case Study 3: Double Biomechanical Coupling Leakage
* **Symptom**: High false-positive rates on patients with isolated ACL tears without associated bony or meniscal injuries.
* **Forensic Trace**:
  * In Stage 4 (`V45`), `submission.csv` was written with:
    $$\text{Contusion}' = 0.97 \cdot \text{Contusion} + 0.03 \cdot \text{ACL}$$
    $$\text{Lateral Meniscus}' = 0.98 \cdot \text{Lateral Meniscus} + 0.02 \cdot \text{ACL}$$
  * In Stage 6 (`_v53_sovereign.py`), the script loaded `submission.csv` as an anchor and re-applied:
    $$\text{Contusion}'' = 0.97 \cdot \text{Contusion}' + 0.03 \cdot \text{ACL}'$$
    $$\text{Lateral Meniscus}'' = 0.98 \cdot \text{Lateral Meniscus}' + 0.02 \cdot \text{ACL}'$$
  This resulted in a cumulative $+6.0\%$ and $+4.0\%$ leakage, artificially inflating contusion and meniscus predictions on pure isolated ACL sprains.
* **Permanent Invariant**: Orthogonal 12x12 coupling is computed strictly once at the final output boundary.

---

### Case Study 4: Auxiliary 16-Window Feature Dilution
* **Symptom**: Integrating auxiliary MIL models trained on 16-window volumes (`all_vols.npy`) at weights $W \ge 0.15$ monotonically degraded public leaderboard performance:
  * $W = 0.50 \to \text{AUC } 0.9530$
  * $W = 0.35 \to \text{AUC } 0.9540$
  * $W = 0.20 \to \text{AUC } 0.9550$
  * $W = 0.00 \to \text{AUC } 0.9580$
* **Root Cause**: The 16-window representations lacked the through-plane spatial resolution of our 528-window localized OAI models. High-weight blending diluted the sharp, localized features learned by the OAI-pretrained backbones.
* **Permanent Invariant**: Pinned auxiliary reader weights to exact calibrated ranges ($0.003\text{ to }0.035$, mean $0.016$), preserving OAI dominance while capturing residual feature diversity.

---

### Case Study 5: 4-vCPU Virtual Machine GIL & Thread Contention (V58)
* **Symptom**: Candidate V58 attempted to optimize runtime by increasing `ThreadPoolExecutor(max_workers=12)` and lowering `chunk_size` from 16 to 8. The run took **660.9 seconds**, an increase of $+109.2\text{ seconds}$ over V56 ($551.7\text{ s}$), exceeding the 500-second Efficiency Prize limit.
* **Forensic Profiling**:
  * Kaggle's T4 x2 execution environment allocates only **4 vCPUs**.
  * Spawning 12 concurrent worker threads caused severe operating system context-switching overhead and Python Global Interpreter Lock (GIL) thrashing.
  * Halving `chunk_size` from 16 to 8 doubled the number of CUDA kernel launch dispatches, saturating the GPU command queue.
* **Permanent Invariant**: Locked thread pool concurrency strictly to 6 workers, and preserved `chunk_size = 16` for optimal tensor batching.

---

### Case Study 6: Kaggle 20 GB Mount Quota Container Crash
* **Symptom**: Attempting to attach 8 separate FP32 fold weight datasets caused Kaggle container initialization to crash immediately with `FileNotFoundError`, leaving `/kaggle/input` completely empty.
* **Root Cause**: Kaggle enforces a hard quota of approximately **20 GB** total attached dataset volume. The combined dataset size:
  * `knee-raptor-corpus`: $15.90\text{ GB}$
  * `knee-raptor-corpus-ext`: $3.50\text{ GB}$
  * 8 separate FP32 fold datasets: $4.55\text{ GB}$
  * Total: $23.95\text{ GB} > 20.0\text{ GB}$ limit.
* **Permanent Invariant**: Developed `tools/soup_weights.py` to merge multi-fold weights into a Weight-Space Model Soup and cast tensors to `FP16` (`torch.half()`), collapsing $4.55\text{ GB}$ across 8 datasets into a single **$559\text{ MB}$** bundle (`cyruspinto/rsna-knee-v50-sovereign-bundle`).
