import pandas as pd
import numpy as np

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

def main():
    dfs = [pd.read_csv(f'scratch/ntejas_repo/oof/armA_oof_fold{i}.csv') for i in range(5)]
    armA_full = pd.concat(dfs, ignore_index=True).rename(columns={'STUDY_UID': 'StudyInstanceUID'})
    print(f"Arm A total rows: {len(armA_full)} | Unique studies: {armA_full['StudyInstanceUID'].nunique()}")

    labels = pd.read_parquet('data/SUPER_CONSENSUS_labels.parquet')
    m = pd.merge(labels, armA_full, on='StudyInstanceUID', suffixes=('_gt', '_pred'))
    print(f"Merged validation count with Super-Consensus: {len(m)}")

    classes = [
        'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
        'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
        'Synovitis', "Baker's", 'Contusion', 'Fracture'
    ]

    aucs = []
    print(f"{'Class':<20} {'Full 5-Fold AUC':<16} {'Positives':<12}")
    print("-" * 50)
    for c in classes:
        y_true = (m[c + '_gt'] >= 0.5).astype(int).values
        y_pred = m[c + '_pred'].values
        auc = exact_roc_auc(y_true, y_pred)
        aucs.append(auc)
        print(f"{c:<20} {auc:<16.4f} {y_true.sum():<5} / {len(y_true)}")

    macro = np.mean(aucs)
    print("-" * 50)
    print(f"{'ARM A FULL 5-FOLD MACRO':<20} {macro:<16.5f}")

if __name__ == '__main__':
    main()
