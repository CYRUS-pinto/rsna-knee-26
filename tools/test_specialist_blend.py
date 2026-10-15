"""
Specialist Blending Verification: Tests whether surgical column-specific injection
of Engine 4 on its strongest findings (Synovitis 0.82, Effusion 0.76) improves OOF performance
without dragging down acute ligament tears.
"""

import pandas as pd
import numpy as np

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def exact_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true, bool)
    y_score = np.asarray(y_score, float)
    pos = y_score[y_true]
    neg = y_score[~y_true]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    all_scores = np.concatenate([pos, neg])
    ranks = pd.Series(all_scores).rank(method='average').to_numpy()
    sum_ranks_pos = ranks[:n_pos].sum()
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))

def rankpct(df):
    return df.rank(pct=True, method='average')

oof_e4 = pd.read_parquet('models/engine4_kaggle_output/oof_engine4_fold0.parquet')
super_df = pd.read_parquet('data/SUPER_CONSENSUS_labels.parquet')

merged = oof_e4.merge(super_df, on='StudyInstanceUID', suffixes=('_e4', '_true'))
print(f"Validation cohort: {len(merged)} studies")

# Benchmark Engine 4 solo
aucs_e4 = {}
for col in LABELS:
    yt = merged[f"{col}_true"] >= 0.5
    yp = merged[f"{col}_e4"]
    aucs_e4[col] = exact_roc_auc(yt, yp)

print("Engine 4 Solo Top Strengths:")
for col in sorted(aucs_e4, key=aucs_e4.get, reverse=True)[:5]:
    print(f"  {col:20s}: {aucs_e4[col]:.4f}")

print("\nEngine 4 Weaknesses (Must have ZERO weight):")
for col in sorted(aucs_e4, key=aucs_e4.get)[:4]:
    print(f"  {col:20s}: {aucs_e4[col]:.4f}")
