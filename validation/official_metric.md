# Official Competition Metric Documentation (`official_metric.py` & `gate_score.py`)

## 1. Overview
The RSNA Knee Abnormality Detection competition evaluates performance using **Macro-Averaged Area Under the Receiver Operating Characteristic Curve (Macro-AUC)** across all 12 target pathologies:

$$\text{Competition Score} = \frac{1}{12} \sum_{j=1}^{12} \text{ROC-AUC}_j$$

---

## 2. Mathematical Definition: Mann-Whitney U Statistic
The Area Under the ROC Curve for a binary ground-truth vector $\mathbf{y} \in \{0, 1\}^N$ and continuous prediction scores $\mathbf{s} \in \mathbb{R}^N$ is mathematically equivalent to the normalized Mann-Whitney U rank-sum test:

$$\text{AUC} = \frac{\sum_{i \in \mathcal{P}} \text{Rank}(s_i) - \frac{N_+ (N_+ + 1)}{2}}{N_+ \cdot N_-}$$

where:
- $\mathcal{P} = \{i : y_i = 1\}$ is the positive set of size $N_+$.
- $\mathcal{N} = \{i : y_i = 0\}$ is the negative set of size $N_-$.
- $\text{Rank}(s_i)$ is the fractional average rank of prediction score $s_i$ in the joint sorted array of all $N = N_+ + N_-$ predictions.

### Key Property: Invariant to Monotonic Transformations
ROC-AUC depends **strictly on the relative order of predictions**, not their absolute calibration:
$$\text{AUC}(\mathbf{y}, \mathbf{s}) = \text{AUC}(\mathbf{y}, f(\mathbf{s}))$$
for any strictly monotonically increasing function $f(x)$ (such as sigmoid, log-odds, temperature scaling, or percentile ranking).

---

## 3. Critical Implementation Safeguards in `official_metric.py`

### 3.1 Raw Logit Scoring vs. Sigmoid Saturation
In standard PyTorch pipelines, predictions are often converted via `torch.sigmoid(logits)`. However, floating-point precision causes extreme logits (e.g. $> 16.0$ or $< -16.0$) to saturate at `1.0` or `0.0` in FP16 or FP32.
When multiple samples share identical saturated prediction values:
- Average ranking assigns them tied ranks.
- The discriminator loses ordering resolution.
- ROC-AUC suffers an artificial drop of $0.0005 - 0.0020$.

`official_metric.py` and `gate_score.py` enforce evaluating continuous **raw logits** or strictly double-precision continuous values before any rounding.

### 3.2 Degenerate Class Guardrail
If a validation split contains zero positive ($N_+ = 0$) or zero negative ($N_- = 0$) instances for an exceptionally rare finding, standard libraries raise an exception or divide by zero.
`exact_roc_auc` handles this gracefully:
```python
if n_pos == 0 or n_neg == 0:
    return 0.5  # Random guessing baseline
```

---

## 4. Usage in Validation Pipeline
```bash
# Score out-of-fold predictions against ground truth
python validation/gate_score.py --pred oof_predictions.parquet --truth data/train_folds.parquet
```
The script outputs:
1. Per-class ROC-AUC for each of the 12 findings.
2. Macro-average AUC with 5-decimal precision.
3. Class imbalance and prediction distribution statistics.
