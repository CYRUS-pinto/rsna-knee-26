"""
Offline Paired Bootstrap Cross-Validation Harness
RSNA Knee Abnormality Detection Challenge 2026

Calculates macro-averaged ROC-AUC with paired bootstrap hypothesis testing
to distinguish genuine architectural gains from random public leaderboard noise.
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def fast_roc_auc(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Pure NumPy Mann-Whitney U calculation of ROC-AUC.
    Zero external dependencies required.
    """
    n1 = np.sum(y_true == 1)
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(y_pred).rank(method='average').values
    r1 = np.sum(ranks[y_true == 1])
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def compute_macro_auc(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes macro-averaged ROC-AUC across all 12 targets.
    """
    aucs = []
    for j in range(y_true.shape[1]):
        col_true = y_true[:, j]
        col_pred = y_pred[:, j]
        val = fast_roc_auc(col_true, col_pred)
        if not np.isnan(val):
            aucs.append(val)
    return float(np.mean(aucs)) if len(aucs) > 0 else 0.50

def run_paired_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 1000, seed: int = 42):
    np.random.seed(seed)
    n = len(y_true)
    
    auc_a_base = compute_macro_auc(y_true, pred_a)
    auc_b_base = compute_macro_auc(y_true, pred_b)
    delta_base = auc_b_base - auc_a_base
    
    deltas = np.zeros(n_boot)
    for i in range(n_boot):
        boot_idx = np.random.choice(n, n, replace=True)
        boot_y = y_true[boot_idx]
        boot_a = pred_a[boot_idx]
        boot_b = pred_b[boot_idx]
        deltas[i] = compute_macro_auc(boot_y, boot_b) - compute_macro_auc(boot_y, boot_a)
        
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    p_val = np.mean(deltas <= 0.0) if delta_base > 0 else np.mean(deltas >= 0.0)
    
    return {
        'auc_a': auc_a_base,
        'auc_b': auc_b_base,
        'delta': delta_base,
        'ci_95': (ci_low, ci_high),
        'p_value': p_val,
        'n_samples': n
    }

def main():
    parser = argparse.ArgumentParser(description="Paired Bootstrap Evaluation Harness")
    parser.add_argument("--pred_a", type=str, required=True, help="Path to baseline prediction CSV (e.g. V17)")
    parser.add_argument("--pred_b", type=str, required=True, help="Path to candidate prediction CSV (e.g. V31)")
    parser.add_argument("--truth", type=str, default="data/train_folds.parquet", help="Path to ground truth parquet/csv")
    parser.add_argument("--gold_only", action="store_true", help="Evaluate strictly on verified gold studies")
    parser.add_argument("--n_boot", type=int, default=1000, help="Number of bootstrap iterations")
    args = parser.parse_args()
    
    # Load truth
    truth_path = Path(args.truth)
    if truth_path.suffix == '.parquet':
        df_truth = pd.read_parquet(truth_path)
    else:
        df_truth = pd.read_csv(truth_path)
        
    if args.gold_only and 'is_gold' in df_truth.columns:
        df_truth = df_truth[df_truth['is_gold'] == True]
        print(f"Filtered strictly to {len(df_truth)} Gold ground truth studies.")
        
    df_truth = df_truth.set_index('StudyInstanceUID')
    
    # Load predictions
    df_a = pd.read_csv(args.pred_a).set_index('StudyInstanceUID')
    df_b = pd.read_csv(args.pred_b).set_index('StudyInstanceUID')
    
    # Common study intersection
    common = df_truth.index.intersection(df_a.index).intersection(df_b.index)
    if len(common) == 0:
        print("Error: Zero intersecting studies between ground truth and predictions!", file=sys.stderr)
        sys.exit(1)
        
    print(f"Benchmarking on {len(common)} intersecting studies with {args.n_boot} bootstrap iterations...")
    
    y_true = df_truth.loc[common, TARGETS].values
    pred_a = df_a.loc[common, TARGETS].values
    pred_b = df_b.loc[common, TARGETS].values
    
    # Binarize truth if needed (threshold at 0.5 for soft labels)
    y_true_binary = (y_true >= 0.5).astype(int)
    
    res = run_paired_bootstrap(y_true_binary, pred_a, pred_b, n_boot=args.n_boot)
    
    print("\n" + "="*60)
    print("PAIRED BOOTSTRAP EVALUATION RESULTS")
    print("="*60)
    print(f"Model A (Baseline):  AUC = {res['auc_a']:.4f}")
    print(f"Model B (Candidate): AUC = {res['auc_b']:.4f}")
    print(f"Delta (B - A):       {res['delta']:+.4f}")
    print(f"95% Confidence Int:  [{res['ci_95'][0]:+.4f}, {res['ci_95'][1]:+.4f}]")
    print(f"Paired p-value:      p = {res['p_value']:.4f}")
    
    if res['p_value'] < 0.05 and res['delta'] > 0:
        print("\nDECISION: SIGNIFICANT IMPROVEMENT (Passes local displacement gate)")
    elif res['p_value'] < 0.05 and res['delta'] < 0:
        print("\nDECISION: SIGNIFICANT REGRESSION (Rejected)")
    else:
        print("\nDECISION: STATISTICALLY INDISTINGUISHABLE (Noise threshold, do not displace)")
    print("="*60)

if __name__ == '__main__':
    main()
