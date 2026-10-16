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

def rank_transform(s):
    ranks = pd.Series(s).rank(method='average', pct=True).to_numpy()
    return ranks

def main():
    labels = pd.read_parquet('data/SUPER_CONSENSUS_labels.parquet')
    oof5 = pd.read_parquet('training/engine5_decoupled/kaggle_output/oof/oof_engine5_combined_fold0.parquet')
    arm_a = pd.read_csv('scratch/ntejas_repo/oof/armA_oof_fold0.csv')
    arm_a = arm_a.rename(columns={'STUDY_UID': 'StudyInstanceUID'})

    classes = [
        'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
        'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
        'Synovitis', "Baker's", 'Contusion', 'Fracture'
    ]

    # Merge on StudyInstanceUID
    m = pd.merge(labels, oof5, on='StudyInstanceUID', suffixes=('_gt', '_e5'))
    m = pd.merge(m, arm_a, on='StudyInstanceUID')
    print(f"Total overlapping validation studies between GroundTruth, Engine5, and ArmA: {len(m)}")

    print(f"{'Class':<20} {'ArmA AUC':<12} {'Engine5 AUC':<14} {'Simple Avg':<12} {'Rank Blend':<12} {'Best Weight(E5)':<15}")
    print("-" * 85)

    arma_aucs = []
    e5_aucs = []
    avg_aucs = []
    rank_aucs = []
    opt_aucs = []
    best_weights = {}

    for c in classes:
        y_true = (m[c + '_gt'] >= 0.5).astype(int).values
        pred_arma = m[c].values
        pred_e5 = m[c + '_e5'].values

        auc_arma = exact_roc_auc(y_true, pred_arma)
        auc_e5 = exact_roc_auc(y_true, pred_e5)

        # Simple average
        pred_avg = 0.5 * pred_arma + 0.5 * pred_e5
        auc_avg = exact_roc_auc(y_true, pred_avg)

        # Rank blend
        r_arma = rank_transform(pred_arma)
        r_e5 = rank_transform(pred_e5)
        pred_rank = 0.5 * r_arma + 0.5 * r_e5
        auc_rank = exact_roc_auc(y_true, pred_rank)

        # Grid search best weight for Engine 5
        best_w = 0.0
        best_auc = auc_arma
        for w in np.linspace(0.0, 1.0, 21):
            blended = (1 - w) * r_arma + w * r_e5
            auc_test = exact_roc_auc(y_true, blended)
            if auc_test > best_auc:
                best_auc = auc_test
                best_w = w

        arma_aucs.append(auc_arma)
        e5_aucs.append(auc_e5)
        avg_aucs.append(auc_avg)
        rank_aucs.append(auc_rank)
        opt_aucs.append(best_auc)
        best_weights[c] = (best_w, best_auc - auc_arma)

        print(f"{c:<20} {auc_arma:<12.4f} {auc_e5:<14.4f} {auc_avg:<12.4f} {auc_rank:<12.4f} w={best_w:.2f} ({best_auc-auc_arma:+.4f})")

    print("-" * 85)
    print(f"{'OVERALL MACRO':<20} {np.mean(arma_aucs):<12.4f} {np.mean(e5_aucs):<14.4f} {np.mean(avg_aucs):<12.4f} {np.mean(rank_aucs):<12.4f} {np.mean(opt_aucs):<12.4f}")

if __name__ == '__main__':
    main()
