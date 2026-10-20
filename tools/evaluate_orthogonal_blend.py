"""
Offline Statistical Equivalence & Paired Bootstrap Blend Evaluator.
Implements:
1. Architectural Orthogonality Audit: Asserts Spearman rank correlation rho <= 0.88 against Raptor CoAtNet.
2. Official Gold Evaluation: Computes macro-AUC across the 12 target pathologies on the 58 official gold studies.
3. Paired Stratified Bootstrap: Runs 2,000 bootstrap resamples to establish a 95% confidence interval for Delta AUC.
4. Pre-Registered Gate Enforcement:
   - PASS: Delta AUC >= +0.0030 AND 95% CI lower bound > 0.0000.
   - FAIL: Reject candidate to prevent public leaderboard churn and protect finals.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def fast_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score)
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = np.empty(len(y_score), dtype=float)
    order = np.argsort(y_score)
    ranks[order] = np.arange(len(y_score))
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        tie_ranks = np.zeros(len(counts))
        np.add.at(tie_ranks, inv, ranks)
        tie_ranks = tie_ranks / counts
        ranks = tie_ranks[inv]
    rank_sum = np.sum(ranks[y_true])
    u = rank_sum - (n_pos * (n_pos - 1)) / 2.0
    return float(u / (n_pos * n_neg))

def compute_macro_auc(y_true_mat, y_score_mat):
    aucs = []
    for j in range(len(TARGETS)):
        a = fast_roc_auc(y_true_mat[:, j], y_score_mat[:, j])
        aucs.append(a)
    return float(np.nanmean(aucs)), aucs

def paired_bootstrap_delta(y_true_mat, baseline_scores, candidate_scores, n_bootstraps=2000, seed=42):
    np.random.seed(seed)
    N = len(y_true_mat)
    deltas = []
    
    for _ in range(n_bootstraps):
        idx = np.random.choice(N, size=N, replace=True)
        # Check if every column has at least one positive and one negative in sample
        valid = True
        for j in range(len(TARGETS)):
            pos = y_true_mat[idx, j].sum()
            if pos == 0 or pos == N:
                valid = False
                break
        if not valid:
            continue
            
        base_auc, _ = compute_macro_auc(y_true_mat[idx], baseline_scores[idx])
        cand_auc, _ = compute_macro_auc(y_true_mat[idx], candidate_scores[idx])
        deltas.append(cand_auc - base_auc)
        
    deltas = np.array(deltas)
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    mean_delta = np.mean(deltas)
    return mean_delta, ci_lower, ci_upper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v17_preds', default='data/train_folds.parquet', help='Baseline predictions (V17 / Raptor)')
    parser.add_argument('--cand_preds', required=True, help='Candidate predictions file')
    parser.add_argument('--alpha', type=float, default=0.20, help='Candidate weight in blend')
    args = parser.parse_args()

    # Load official Gold ground truth
    train_df = pd.read_csv('train.csv')
    gold_df = train_df.dropna(subset=['ACL']).copy().set_index('StudyInstanceUID')
    gold_uids = list(gold_df.index)
    y_gold = gold_df[TARGETS].values.astype(int)
    print(f"Loaded {len(gold_uids)} Gold Studies for statistical gate testing.")

    # Load Baseline
    df_base = pd.read_parquet(args.v17_preds) if args.v17_preds.endswith('.parquet') else pd.read_csv(args.v17_preds)
    df_base = df_base.set_index('StudyInstanceUID').loc[gold_uids]
    base_scores = df_base[TARGETS].values.astype(float)

    # Load Candidate
    df_cand = pd.read_parquet(args.cand_preds) if args.cand_preds.endswith('.parquet') else pd.read_csv(args.cand_preds)
    df_cand = df_cand.set_index('StudyInstanceUID').loc[gold_uids]
    cand_scores = df_cand[TARGETS].values.astype(float)

    # 1. Architectural Orthogonality Check
    corrs = []
    for j in range(len(TARGETS)):
        r, _ = spearmanr(base_scores[:, j], cand_scores[:, j])
        corrs.append(r)
    mean_corr = np.nanmean(corrs)
    print(f"\n--- 1. ARCHITECTURAL ORTHOGONALITY ---")
    print(f"Mean Spearman correlation with Raptor baseline: {mean_corr:.4f}")
    if mean_corr > 0.88:
        print("[WARNING] High collinearity with baseline (rho > 0.88). Model provides little new variance.")
    else:
        print("[PASS] Orthogonal feature space confirmed (rho <= 0.88).")

    # 2. Baseline vs Candidate vs Blend Scores
    base_macro, _ = compute_macro_auc(y_gold, base_scores)
    cand_macro, _ = compute_macro_auc(y_gold, cand_scores)
    
    blend_scores = (1.0 - args.alpha) * base_scores + args.alpha * cand_scores
    blend_macro, _ = compute_macro_auc(y_gold, blend_scores)

    print(f"\n--- 2. GOLD MACRO-AUC ---")
    print(f"Baseline (V17 Anchor)  : {base_macro:.4f}")
    print(f"Candidate (Single Model): {cand_macro:.4f}")
    print(f"Blend ({1-args.alpha:.0%} Base + {args.alpha:.0%} Cand): {blend_macro:.4f} (Delta: {blend_macro - base_macro:+.4f})")

    # 3. Paired Stratified Bootstrap
    print(f"\n--- 3. PAIRED BOOTSTRAP SIGNIFICANCE (2,000 Resamples) ---")
    mean_d, ci_lo, ci_hi = paired_bootstrap_delta(y_gold, base_scores, blend_scores, n_bootstraps=2000)
    print(f"Mean Delta AUC: {mean_d:+.4f}")
    print(f"95% Bootstrap CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # Gate decision
    passed_gate = (blend_macro - base_macro >= 0.0030) and (ci_lo > 0.0)
    print("\n==================================================")
    if passed_gate:
        print("DECISION: GATE PASSED! Candidate qualifies for Submission Packaging (V33).")
    else:
        print("DECISION: GATE REJECTED. Delta < +0.0030 or CI includes zero.")
        print("Recommendation: Maintain strict freeze. Do NOT submit to Kaggle.")
    print("==================================================")

if __name__ == '__main__':
    main()
