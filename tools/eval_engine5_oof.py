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
    labels = pd.read_parquet('data/SUPER_CONSENSUS_labels.parquet')
    oof5 = pd.read_parquet('training/engine5_decoupled/kaggle_output/oof/oof_engine5_combined_fold0.parquet')

    merged = pd.merge(labels, oof5, on='StudyInstanceUID', suffixes=('_true', '_pred'))
    print(f"Merged validation count: {len(merged)}")

    classes = [
        'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
        'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
        'Synovitis', "Baker's", 'Contusion', 'Fracture'
    ]

    print(f"{'Class':<22} {'AUC':<10} {'Positives':<12}")
    print("-" * 50)
    aucs = {}
    for c in classes:
        y_true = (merged[c + '_true'] >= 0.5).astype(int)
        y_pred = merged[c + '_pred'].values
        auc = exact_roc_auc(y_true, y_pred)
        aucs[c] = auc
        print(f"{c:<22} {auc:<10.4f} {y_true.sum():<5} / {len(y_true)}")

    macro_auc = np.mean(list(aucs.values()))
    print("-" * 50)
    print(f"{'ENGINE 5 FOLD 0 MACRO':<22} {macro_auc:<10.5f}")

    # Off-diagonal correlation matrix check
    pred_matrix = merged[[c + '_pred' for c in classes]].values
    corr = np.corrcoef(pred_matrix, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    mean_off_diag = np.abs(corr).sum() / (12 * 11)
    max_off_diag = np.abs(corr).max()
    print(f"Mean Off-Diagonal Absolute Correlation: {mean_off_diag:.4f}")
    print(f"Max Off-Diagonal Correlation:          {max_off_diag:.4f}")

    corr_df = pd.DataFrame(corr, index=classes, columns=classes)
    print("\nPairs with absolute correlation > 0.60:")
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            val = corr_df.iloc[i, j]
            if abs(val) > 0.60:
                print(f"  {classes[i]:<20} <-> {classes[j]:<20}: {val:.4f}")

if __name__ == '__main__':
    main()
