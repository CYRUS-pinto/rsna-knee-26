"""
Extract and Format Engine 6 Out-Of-Fold (OOF) Predictions for User Audit.
Runs Gates 1 & 2:
Gate 1: Collapse Check (Cross-correlation matrix of 12 prediction columns; flags if r > 0.85)
Gate 2: Fold Macro-AUC score against ground truth labels (Target: >= 0.88, Drop if < 0.85)
Outputs clean artifacts/models/engine6/engine6_oof_predictions.csv for user verification.
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd

ALL_LABELS = [
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

def audit_oof(oof_csv_path, ground_truth_path=None):
    print("=" * 80)
    print(f"AUDITING ENGINE 6 OOF PREDICTIONS: {oof_csv_path}")
    print("=" * 80)
    
    df_oof = pd.read_csv(oof_csv_path)
    assert 'StudyInstanceUID' in df_oof.columns, "Missing StudyInstanceUID"
    cols = [c for c in ALL_LABELS if c in df_oof.columns]
    
    # ----------------- GATE 1: COLLAPSE CHECK -----------------
    print("\n>>> GATE 1: COLLAPSE & CROSS-CORRELATION CHECK <<<")
    corr_matrix = df_oof[cols].corr()
    print(corr_matrix.round(3))
    
    high_corr_pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.85:
                high_corr_pairs.append((cols[i], cols[j], r))
                
    if high_corr_pairs:
        print(f"\n[WARNING] Found high inter-class correlation (r > 0.85): {high_corr_pairs}")
        print("Gate 1 Status: FAIL / SUSPECT (Representation Collapse Detected)")
        gate1_pass = False
    else:
        print("\n[SUCCESS] No pathological inter-class correlation found (all r <= 0.85).")
        print("Gate 1 Status: PASS (Independent Finding Representations Confirmed)")
        gate1_pass = True

    # ----------------- GATE 2: FOLD AUC CHECK -----------------
    gate2_pass = False
    if ground_truth_path and os.path.exists(ground_truth_path):
        print("\n>>> GATE 2: GROUND TRUTH MACRO-AUC VERIFICATION <<<")
        df_gt = pd.read_parquet(ground_truth_path) if ground_truth_path.endswith('.parquet') else pd.read_csv(ground_truth_path)
        merged = df_oof.merge(df_gt, on='StudyInstanceUID', suffixes=('_pred', '_true'))
        
        aucs = {}
        for col in cols:
            y_t = merged[f"{col}_true"].values
            y_p = merged[f"{col}_pred"].values
            aucs[col] = exact_roc_auc(y_t, y_p)
            print(f"  {col:<18}: {aucs[col]:.4f}")
            
        macro_auc = float(np.mean(list(aucs.values())))
        print("-" * 80)
        print(f"ENGINE 6 FOLD MACRO-AUC: {macro_auc:.4f}")
        print("-" * 80)
        
        if macro_auc >= 0.88:
            print(f"Gate 2 Status: PASS (AUC {macro_auc:.4f} >= 0.88 - Competitive with R6/Raptor)")
            gate2_pass = True
        elif macro_auc >= 0.85:
            print(f"Gate 2 Status: MARGINAL (AUC {macro_auc:.4f} in [0.85, 0.88) - Proceed with Caution)")
            gate2_pass = True
        else:
            print(f"Gate 2 Status: FAIL (AUC {macro_auc:.4f} < 0.85 - Engine 4 style collapse)")
            gate2_pass = False

    return gate1_pass, gate2_pass

if __name__ == '__main__':
    if len(sys.argv) > 1:
        audit_oof(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print("Usage: python tools/extract_engine6_oof.py <path_to_oof_csv> [path_to_ground_truth]")
