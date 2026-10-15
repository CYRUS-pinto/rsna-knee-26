# Tools Directory (`tools/`)

This directory contains offline validation gates, dataset extraction utilities, and official metric computation scripts used to prevent leaderboard regressions and evaluate candidate models before Kaggle submissions.

## Index of Tools

| Tool | Purpose | Companion Docs |
| :--- | :--- | :--- |
| [`gate_make_root.py`](gate_make_root.py) | Filters and structures the 58 Gold studies from `train.csv` into an isolated offline evaluation root. | [`gate_make_root.md`](gate_make_root.md) |
| [`gate_score.py`](gate_score.py) | Runs paired bootstrap significance tests ($N=2,000$) across inference arms to enforce the zero-regression gate. | [`gate_score.md`](gate_score.md) |
| [`official_metric.py`](official_metric.py) | Standalone implementation of the RSNA 2026 Macro-Averaged ROC-AUC competition metric. | [`official_metric.md`](../validation/official_metric.md) |

## The Offline Gold Gate Protocol
To prevent burning daily Kaggle submissions on noisy or regressive hypotheses, any new model weight or ensemble configuration must pass through the Gold-58 Gate:
1. `python tools/gate_make_root.py --data-dir data/ --output-dir gate_root/`
2. Run inference pipeline on `gate_root/`
3. `python tools/gate_score.py --gold gate_root/gold.csv --arm engine1 engine2 engine3`
4. **Promotion Criterion**: Candidate must strictly beat baseline in paired bootstrap at $95\%$ confidence interval ($[\text{CI}_{\text{low}}, \text{CI}_{\text{high}}] > 0$) with zero study fallbacks.
