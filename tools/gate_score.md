# `tools/gate_score.py` Architectural Documentation

## Purpose
`gate_score.py` executes paired bootstrap statistical hypothesis testing ($N = 2,000$ resamples) to evaluate whether a new model arm or ensemble variation produces a statistically significant improvement over baseline, preventing regressions caused by public leaderboard noise.

## Mathematical Methodology
1. **Per-Class ROC-AUC**: Calculates AUC for each of the 12 clinical pathologies on the Gold-58 evaluation split.
2. **Macro-Average AUC**: Averages valid pathology AUCs:
   $$\text{Macro-AUC} = \frac{1}{|C|} \sum_{c \in C} \text{AUC}_c$$
3. **Paired Bootstrap Difference**:
   $$\Delta_b = \text{Macro-AUC}_b(\text{Candidate}) - \text{Macro-AUC}_b(\text{Baseline}), \quad b = 1, \dots, 2000$$
4. **Admissibility Gate**:
   - The 95% empirical bootstrap confidence interval $[\text{CI}_{2.5\%}, \text{CI}_{97.5\%}]$ must have $\text{CI}_{2.5\%} > 0$.
   - Inter-arm Pearson correlation must satisfy $\rho \ge 0.30$.
   - Study fallbacks must equal exactly 0.

## Usage
```bash
python tools/gate_score.py \
  --gold gate_root/gold.csv \
  --baseline baseline_preds.csv \
  --candidate candidate_preds.csv \
  --n-boot 2000
```
