# ADR-035: Silver Defense, Empirical Run Time Deconstruction, and the Unconfounded Slot-2 Protocol

## Status
**APPROVED & ACTIVE** — October 17, 2026 (5 Days Remaining Until Deadline)

---

## 1. Context & Objective Shift: The Medal Reality

A direct query to the Kaggle competition API confirmed:
* `awardsPoints: True`
* `reward: $77,000 USD`
* `teamCount: 6,146 teams`

### Official Medal Cutoffs (Standard Kaggle Tier Thresholds):
* **Gold Cutoff:** Top 10 + 0.2% of teams = $\lfloor 10 + 0.002 \times 6146 \rfloor =$ **Top 22 teams** ($\approx 0.965+$ AUC).
* **Silver Cutoff:** Top 5% of teams = $\lfloor 0.05 \times 6146 \rfloor =$ **Top 307 teams** ($\approx 0.958$ AUC).
* **Bronze Cutoff:** Top 10% of teams = $\lfloor 0.10 \times 6146 \rfloor =$ **Top 614 teams** ($\approx 0.956$ AUC).

### Benchmark Standing & Leaderboard Density:
* **Current Personal Best:** **0.958** (Candidate V17 / V20, 23 total entries).
* **Leaderboard Density:** With approximately **350–400 teams per 0.001 AUC**, private leaderboard shakeout of $\pm 0.0015$ can swing a team hundreds of places in either direction.
* **Medal Reality:** Medals are decided strictly on the private leaderboard (70% of the test set). Silver is roughly a coin flip. No single submission "guards" it.
* **Strategic Objective:** Do not chase high-frequency public noise. Deploy two robust candidate models (Slot 1 = V17, Slot 2 = V6) to provide a mild ConvNeXt checkpoint hedge (R6 vs. R7) alongside the proven Raptor arm.

---

## 2. Empirical Run Time Deconstruction: Commit vs. Evaluation Reality

Historical notes cited "438s" for Candidate V6 and suggested it could serve as an "Efficiency Prize Hedge." Direct examination of downloaded Kaggle logs and submission timestamps establishes the ground truth:

### Downloaded Kernel Commit Logs Analysis:
1. **Candidate V17 (`cyruspinto/rsna-knee-v17-multiscale-r6-fliptta-blend`):**
   * Commit run time on 3 mock studies: **443.4s**.
   * Breakdown: Raptor 384: 46.6s | Raptor 224: 24.8s | ConvNeXt R6: 340.1s.
2. **Candidate V28 (`cyruspinto/rsna-knee-v28-calibrated-grandmaster-master`):**
   * Commit run time on 3 mock studies: **878.6s** (total container 893.3s).
   * Pip install & setup overhead: 552.5s | ConvNeXt R7: 325.1s.
3. **Candidate V26 Fast Raptor (`cyruspinto/rsna-knee-fast-raptor-multiscale`):**
   * Commit run time on 3 mock studies: **111.4s** (total container 119.4s).

### Real Evaluation Durations:
* Timestamp bounds from submission queues indicate real evaluation takes **~3.9 hours** for V17 and **~5.5 hours** for V25 (including queue wait).
* **The "438s" figure was simply the commit run time on 3 mock studies.** V6 and V17 take hours on the real hidden test set.

### Official Efficiency Prize Formula & Slot 2 Resolution:
The official efficiency score is:
$$\text{Efficiency} = \frac{\text{AUC}}{\text{Benchmark} - \max\text{AUC}} + \frac{\text{RuntimeSeconds}}{32400}$$
* With $\text{Benchmark} = 0.500$ and $\max\text{AUC} \approx 0.969$:
  $$\text{Denominator} = 0.500 - 0.969 = -0.469$$
* If a model takes 3.5 hours ($12,600$s), its runtime penalty is:
  $$\frac{12600}{32400} = +0.389$$
* A true efficiency contender (e.g. 15-minute pure fastread) scores $\frac{900}{32400} = +0.028$, beating a 3.5-hour model by **$-0.361$** points on runtime alone.
* **Fatal Realization:** **Candidate V6 is NOT an efficiency model.** It runs for hours on the test set. Submitting V6 as an "efficiency ticket" is mathematically dead on arrival.
* **Decisive Strategic Shift:** **Slot 2 is officially freed from the pseudo-efficiency trap.** Both slots will be deployed to optimize the Main Track and maximize the expected maximum AUC for Silver/Gold.

---

## 3. The Unconfounded Twin Protocol: V32 vs. V32-Ablation

Prior submissions often combined multiple simultaneous changes (e.g., Stride-2 + Non-FS heads + hand-crafted weights), confounding experimental results. To establish clean causality:

1. **Candidate V32 (Full Model):**
   * Stride-2 ConvNeXt R7 + MultiScale Raptor + RadImageNet ResNet-50 E11 Non-FS heads.
2. **Candidate V32-Ablation (Identical Twin):**
   * Exactly identical code, random seeds, and inputs, but with `W_RAD = {c: 0.0 for c in labels}`.
   * Isolates the pure effect of the RadImageNet Non-FS representation from the Stride-2 ConvNeXt speedup.

### Decision Policy for Main Track Promotion:
$$\Delta \text{AUC} = \text{Score}(\text{V32}) - \text{Score}(\text{V32-Ablation})$$
* If $\Delta \text{AUC} \ge +0.002$: Conclusive empirical proof that Non-FS anatomical representation provides meaningful orthogonal signal.
* If $\Delta \text{AUC} < +0.002$: Concludes that either the RadImageNet weights or the Non-FS arm does not justify the added complexity; select the simpler twin.
* **Final Slot Allocations:**
  * **Slot 1:** Candidate V17 (`0.958`, Ref `57253840`) — Verified Silver anchor.
  * **Slot 2:** The best unconfounded performer between V32 and V32-Ablation (or V20 `0.958` if neither improves).

---

## 4. Execution-Grade Code Verification: Zero-Proxy Exec

To avoid false confidence from `ast.parse` (which does not catch `NameError` or undefined runtime symbols) and synthetic proxy scripts (which rewrite code):
* We created [`tools/test_exec_real_cells.py`](../tools/test_exec_real_cells.py).
* It directly extracts the exact cell source strings from the `.ipynb` JSON payloads for both notebooks.
* It executes Cell 10 via Python `exec()` inside an isolated namespace containing synthetic 10-study DataFrames.
* **Results:** Both Cell 10 implementations generated shape `(10, 13)`, finite values bounded in $[0, 1]$, non-zero standard deviations ($> 0.001$), and valid receipts with zero `NameError`s.
* Cell 8's `/tmp/rad_run.py` was patched to initialize `test_ids` and `n_studies` globally and enforce slice-count deterministic tie-breaking.

---

## 5. Kaggle Session & Diagnostic Action

* The stale `QUEUED` V32 session holding up batch execution was cancelled and deleted.
* The 100-Study Diagnostic Kernel ([`scratch/e11_diagnostic_kernel`](../scratch/e11_diagnostic_kernel)) was configured for CPU (`"enable_gpu": false`) and pushed to Kaggle. It runs without consuming GPU quota to evaluate held-out fold head Spearman correlation against negative controls.
* Hard code freeze remains strictly set for **Monday, October 19, 2026**.
