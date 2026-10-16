"""
Constrained Convex Optimizer for Quint-Engine Ensembling.
Finds the global optimal weights (w_raptor, w_convnext, w_engine5, w_engine6)
that maximize the exact official RSNA competition metric (macro-averaged Wilcoxon-Mann-Whitney ROC-AUC).
Uses SciPy SLSQP (Sequential Least Squares Programming) with Dirichlet simplex constraints (sum(w) = 1, w >= 0).
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ALL_LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def exact_roc_auc(y_true, y_score):
    """Zero-dependency exact Wilcoxon-Mann-Whitney ROC-AUC."""
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

def pct_rank(arr):
    return pd.Series(np.asarray(arr, dtype=np.float64)).rank(method='average', pct=True).values

def optimize_class_weights(targets, preds_dict, init_weights=None):
    """
    targets: 1D binary array (N,)
    preds_dict: dict of {engine_name: 1D probability array (N,)}
    """
    engine_names = list(preds_dict.keys())
    m = len(engine_names)
    rank_matrix = np.stack([pct_rank(preds_dict[name]) for name in engine_names], axis=1) # (N, m)
    
    if init_weights is None:
        w0 = np.ones(m) / m
    else:
        w0 = np.asarray(init_weights) / np.sum(init_weights)
        
    def objective(w):
        # We minimize negative macro AUC
        blend = np.dot(rank_matrix, w)
        auc = exact_roc_auc(targets, blend)
        return -auc

    # Constraints: sum(w) = 1.0, 0 <= w_i <= 1.0
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0.0, 1.0) for _ in range(m)]

    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-6})
    best_weights = {name: float(res.x[i]) for i, name in enumerate(engine_names)}
    best_auc = -float(res.fun)
    return best_weights, best_auc

def optimize_quint_blend(y_true_df, oof_dfs):
    """
    Args:
        y_true_df: DataFrame of ground truth labels with StudyInstanceUID
        oof_dfs: dict of {engine_name: DataFrame of OOF predictions}
    """
    print("=" * 80)
    print("OPTIMIZING QUINT-ENGINE ENSEMBLE WEIGHTS VIA SLSQP CONSTRAINED OPTIMIZATION")
    print("=" * 80)
    
    optimal_weights_table = {}
    optimal_aucs = {}
    baseline_aucs = {}

    for col in ALL_LABELS:
        targets = y_true_df[col].values
        preds_dict = {name: df[col].values for name, df in oof_dfs.items()}
        
        # Baseline equal blend
        equal_blend = np.mean([pct_rank(preds_dict[name]) for name in preds_dict], axis=0)
        base_auc = exact_roc_auc(targets, equal_blend)
        baseline_aucs[col] = base_auc
        
        # Optimize weights
        opt_w, opt_auc = optimize_class_weights(targets, preds_dict)
        optimal_weights_table[col] = opt_w
        optimal_aucs[col] = opt_auc
        
        delta = opt_auc - base_auc
        print(f"{col:<18} | Base AUC: {base_auc:.4f} | Opt AUC: {opt_auc:.4f} (+{delta:+.4f}) | Weights: {opt_w}")

    macro_base = np.mean(list(baseline_aucs.values()))
    macro_opt = np.mean(list(optimal_aucs.values()))
    print("=" * 80)
    print(f"OVERALL MACRO AUC: Base = {macro_base:.4f} -> Optimized = {macro_opt:.4f} (+{macro_opt - macro_base:+.4f})")
    print("=" * 80)
    return optimal_weights_table
