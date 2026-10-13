import pandas as pd
import numpy as np
from scipy.stats import rankdata

def compute_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_score)) + 1
    unique_scores, inverse, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    for i, count in enumerate(counts):
        if count > 1:
            ranks[inverse == i] = ranks[inverse == i].mean()
    return (np.sum(ranks[y_true]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

gold_df = pd.read_csv('train.csv')
gold_df = gold_df[gold_df['ACL'].notna()]
FINDINGS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

df_hybrid = pd.read_csv('HYBRID_labels_scores.csv')
df_opus = pd.read_csv('scratch/ntejas_repo/labels_ds/labels_v2_repaired.csv')
df_opus['StudyInstanceUID'] = df_opus['STUDY_UID']

# Merge
m = pd.merge(df_hybrid[['StudyInstanceUID'] + FINDINGS], df_opus[['StudyInstanceUID'] + FINDINGS], on='StudyInstanceUID', suffixes=('_hyb', '_opus'))
m_gold = pd.merge(gold_df[['StudyInstanceUID'] + FINDINGS], m, on='StudyInstanceUID')

print("Testing Blend Weights (HYBRID + Claude Opus 4.5):")
for w_hyb in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
    w_opus = 1.0 - w_hyb
    scores = {}
    for f in FINDINGS:
        # Rank-based blending
        r_hyb = rankdata(m_gold[f'{f}_hyb']) / len(m_gold)
        r_opus = rankdata(m_gold[f'{f}_opus']) / len(m_gold)
        blended_score = w_hyb * r_hyb + w_opus * r_opus
        scores[f] = compute_auc(m_gold[f].values, blended_score)
    macro = np.nanmean(list(scores.values()))
    print(f"Weight {w_hyb:.1f} HYBRID + {w_opus:.1f} OPUS -> Macro ROC AUC: {macro:.4f}")
