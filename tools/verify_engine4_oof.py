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

oof = pd.read_parquet('models/engine4_kaggle_output/oof_engine4_fold0.parquet')
super_df = pd.read_parquet('data/SUPER_CONSENSUS_labels.parquet')

merged = oof.merge(super_df, on='StudyInstanceUID', suffixes=('_pred', '_true'))
print(f"Successfully matched {len(merged)} validation studies.")

aucs = {}
for col in LABELS:
    yt = merged[f"{col}_true"] >= 0.5
    yp = merged[f"{col}_pred"]
    aucs[col] = exact_roc_auc(yt, yp)
    print(f"  - {col:20s}: {aucs[col]:.4f}")

macro = np.mean(list(aucs.values()))
print("=" * 45)
print(f"  EXACT OOF MACRO-AUC: {macro:.5f}")
print("=" * 45)
