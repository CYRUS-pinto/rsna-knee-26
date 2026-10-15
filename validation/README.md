# Validation Suite: Gold-58 Offline Bootstrap Gate & Official Metric

## Overview
A comprehensive offline verification framework designed to prevent leaderboard score regressions and detect edge cases before submitting to Kaggle.

## Components
1. **`official_metric.py`**:
   - Exact mathematical implementation of the competition's Macro-Averaged ROC AUC.
   - Evaluates all 12 classes independently and computes unweighted mean across valid classes.
2. **`gate_make_root.py`**:
   - Isolates the 58 expert-annotated gold studies from `train.csv` into a mock Kaggle submission environment (`gate_root/`).
3. **`gate_score.py`**:
   - 2,000 paired bootstrap iterations comparing model predictions against ground truth.
   - Computes 95% confidence intervals $[\text{CI}_{\text{low}}, \text{CI}_{\text{high}}]$.
   - Rejects candidate pipelines if any target AUC $< 0.60$ or inter-arm correlation $< 0.30$.
