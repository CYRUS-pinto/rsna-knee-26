import pandas as pd
import numpy as np

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

for fn in ['scratch/ntejas_repo/labels_ds/labels_v1_opus45.csv', 'scratch/ntejas_repo/labels_ds/labels_v2_repaired.csv']:
    df = pd.read_csv(fn)
    if 'STUDY_UID' in df.columns:
        df['StudyInstanceUID'] = df['STUDY_UID']
    merged = pd.merge(gold_df[['StudyInstanceUID'] + FINDINGS], df, on='StudyInstanceUID', suffixes=('_gt', '_pred'))
    scores = {}
    for f in FINDINGS:
        col = f'{f}_pred' if f'{f}_pred' in merged.columns else f
        auc = compute_auc(merged[f'{f}_gt'], merged[col])
        scores[f] = auc
    macro = np.nanmean(list(scores.values()))
    print(f'{fn}: Macro ROC AUC = {macro:.4f}')
    for f, s in scores.items():
        print(f'   {f:20s}: {s:.4f}')
