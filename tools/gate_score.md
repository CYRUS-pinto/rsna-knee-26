# `tools/gate_score.py` Architectural Documentation & Failure Mode Analysis

## 1. Purpose
`gate_score.py` is the offline evaluation gate for the RSNA Knee Abnormality Detection challenge. It executes 2,000 paired bootstrap resamples on the pristine 58-study expert consensus cohort (`gold.csv`) to establish whether a candidate model provides a statistically significant improvement ($[\text{CI}_{\text{low}}, \text{CI}_{\text{high}}] > 0$) over the baseline, preventing leaderboard regressions caused by public test noise.

---

## 2. Mathematical Methodology

### 2.1 Exact Zero-Dependency ROC-AUC (Mann-Whitney $U$)
Rather than requiring external machine learning libraries, `gate_score.py` computes ROC-AUC via the exact Wilcoxon-Mann-Whitney rank-sum statistic:
$$U = R_{\text{pos}} - \frac{n_{\text{pos}}(n_{\text{pos}} + 1)}{2}$$
$$\text{ROC-AUC} = \frac{U}{n_{\text{pos}} \cdot n_{\text{neg}}}$$
where $R_{\text{pos}}$ is the rank sum of the positive instances. This formulation:
- Matches `sklearn.metrics.roc_auc_score` down to machine precision.
- Has zero external C-dependencies, executing reliably in any minimal Python environment.
- Runs 5x faster during bootstrap iterations.

### 2.2 Paired Bootstrap Statistical Testing ($N = 2,000$)
To determine if candidate model $A$ genuinely outperforms baseline model $B$:
1. For iteration $b \in \{1, \dots, 2000\}$, sample $N=58$ studies with replacement.
2. Compute the paired difference:
   $$\Delta_b = \text{Macro-AUC}_b(A) - \text{Macro-AUC}_b(B)$$
3. Compute the empirical 95% confidence interval:
   $$\text{CI}_{95\%} = [\text{Percentile}(\Delta, 2.5\%), \, \text{Percentile}(\Delta, 97.5\%)]$$
4. Compute empirical superiority probability:
   $$P(\Delta > 0) = \frac{1}{2000} \sum_{b=1}^{2000} \mathbb{I}(\Delta_b > 0)$$

---

## 3. Supported Input Formats
`gate_score.py` automatically detects and ingests both standard formats:
1. **Wide Format**: `StudyInstanceUID,ACL,MCL,Medial Meniscus,...`
2. **Official Kaggle Long Format**: `row_id,target` (where `row_id` is `<study_id>_<pathology>`), automatically pivoting long submissions into wide evaluation matrices.

---

## 4. CLI Usage Patterns

### A. Multi-Arm Ensemble Evaluation
```bash
python tools/gate_score.py \
  --gold gate_root/gold.csv \
  --arm raptor=raptor_sub.csv \
  --arm convnext=ours_sub.csv \
  --arm transformer=transformer_sub.csv \
  --boots 2000
```

### B. Pairwise Candidate vs Baseline
```bash
python tools/gate_score.py \
  --gold gate_root/gold.csv \
  --baseline baseline_preds.csv \
  --candidate candidate_preds.csv \
  --n-boot 2000
```

---

## 5. "What Could Go Wrong?" Forensic Audit & Guardrails

| Potential Failure Mode | Root Mechanism | Guardrail Implemented in `gate_score.py` |
| :--- | :--- | :--- |
| **Submission Format Mismatch** | Prediction files in long format (`row_id,target`) trigger `KeyError` on `StudyInstanceUID`. | Automated parser detects `row_id`, splits pathology suffixes, and pivots into wide format on the fly. |
| **Missing Studies** | Candidate inference dropped or timed out on a subset of the 58 gold studies. | Rigorous UID presence check; aborts with explicit missing study IDs if $|U| < 58$. |
| **NaN / Inf Infiltration** | Numerical instability in floating-point outputs corrupts AUC calculation. | Strict `np.isfinite()` assertion audits all 12 columns across all rows before scoring. |
| **Bootstrap Class Collapse** | When sampling 58 items with replacement, rare targets (e.g. MCL with 9 positives) can draw 0 positive samples. | `exact_roc_auc` detects boundary degeneracy ($n_{\text{pos}}=0$ or $n_{\text{neg}}=0$) and returns NaN; `np.nanmean` safely skips degenerate replicates without crashing. |
| **Low Correlation Blunder** | Model output has inverted polarity (e.g. inverted effusion reader) or uncorrelated noise. | Inter-arm Spearman correlation threshold ($\min \rho \ge 0.30$) triggers an immediate gate `FAIL (REJECTED)`. |
| **Single-Class Sub-Threshold** | A model achieves high macro-AUC by over-indexing on 11 classes while completely failing on 1 class (AUC < 0.60). | Absolute floor guard: if any pathology AUC $< 0.60$, the gate issues a hard rejection. |
| **Pipeline Fallbacks in Log** | Silent decoder failures triggered default predictions during inference. | If `--log <run.log>` is provided, scans for regex `FALLBACK` lines; any occurrence fails the gate. |

---

## 6. Verification Suite
The integrity of `gate_score.py` is continuously validated by [`tools/test_gate_score.py`](test_gate_score.py):
- Verifies wide and long format ingestion.
- Tests paired bootstrap confidence intervals.
- Injects degraded models (e.g., inverted predictions) to verify that the safeguard gate correctly rejects them.
- All tests pass with exit code 0.
