import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

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

# Look at column names and values in both
print("df_hybrid shape:", df_hybrid.shape)
print("df_opus shape:", df_opus.shape)
print("gold_df shape:", gold_df.shape)

m = pd.merge(df_hybrid[['StudyInstanceUID'] + FINDINGS], df_opus[['StudyInstanceUID'] + FINDINGS], on='StudyInstanceUID', suffixes=('_hyb', '_opus'))
m_gold = pd.merge(gold_df[['StudyInstanceUID'] + FINDINGS], m, on='StudyInstanceUID')

print(f"Total overlapping gold studies: {len(m_gold)}")

hyb_aucs = [compute_auc(m_gold[f], m_gold[f + '_hyb']) for f in FINDINGS]
opus_aucs = [compute_auc(m_gold[f], m_gold[f + '_opus']) for f in FINDINGS]

print(f"\n--- EVALUATION ON {len(m_gold)} GOLD STUDIES ---")
print(f"HYBRID Macro AUC: {np.nanmean(hyb_aucs):.4f}")
print(f"OPUS Macro AUC:   {np.nanmean(opus_aucs):.4f}")
print("-" * 50)
for f, h, o in zip(FINDINGS, hyb_aucs, opus_aucs):
    print(f"{f:<18}: HYB {h:.4f} | OPUS {o:.4f}")
