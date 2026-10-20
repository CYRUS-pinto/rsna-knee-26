"""
Automated Out-Of-Fold (OOF) Nelder-Mead Weight Optimizer.
Solves for the optimal per-pathology ensemble weights:
    max_w ROC-AUC(y_true, sum_m w_m * rank_m)
Subject to:
    sum_m w_m = 1.0, w_m >= 0.0
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def optimize_weights_for_finding(y_true: np.ndarray, model_ranks: list) -> np.ndarray:
    """
    Finds optimal weights for combining M model ranks for a single finding.
    
    Args:
        y_true: Binary ground truth array of shape (N,)
        model_ranks: List of M arrays, each of shape (N,) representing percentile ranks.
        
    Returns:
        optimal_weights: Array of shape (M,) summing to 1.0.
    """
    M = len(model_ranks)
    R = np.column_stack(model_ranks) # (N, M)
    
    # Negative AUC objective
    def objective(raw_weights):
        w = np.exp(raw_weights)
        w = w / np.sum(w)
        blend_rank = R @ w
        try:
            auc = roc_auc_score(y_true, blend_rank)
            return -auc
        except Exception:
            return 0.0
            
    # Initial guess: equal weights
    x0 = np.zeros(M)
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500})
    
    opt_w = np.exp(res.x)
    opt_w = opt_w / np.sum(opt_w)
    return opt_w

def optimize_all_pathologies(df_true: pd.DataFrame, list_of_pred_dfs: list, targets=TARGETS) -> dict:
    """
    Optimizes weights across all 12 pathologies.
    """
    weights_dict = {}
    for col in targets:
        if col not in df_true.columns:
            continue
        y = df_true[col].values
        # Rank-transform each model's predictions
        ranks = []
        for df in list_of_pred_dfs:
            if col in df.columns:
                p = df[col].values
                r = pd.Series(p).rank(method='average', pct=True).values
                ranks.append(r)
            else:
                ranks.append(np.full_like(y, 0.5))
                
        opt_w = optimize_weights_for_finding(y, ranks)
        weights_dict[col] = tuple(np.round(opt_w, 4))
        
    return weights_dict

if __name__ == '__main__':
    N, M = 200, 3
    y = np.random.randint(0, 2, N)
    m1 = y * 0.7 + np.random.randn(N) * 0.3
    m2 = y * 0.5 + np.random.randn(N) * 0.5
    m3 = y * 0.3 + np.random.randn(N) * 0.7
    
    r1 = pd.Series(m1).rank(pct=True).values
    r2 = pd.Series(m2).rank(pct=True).values
    r3 = pd.Series(m3).rank(pct=True).values
    
    w = optimize_weights_for_finding(y, [r1, r2, r3])
    print("[+] Optimized weights for synthetic test:", w)
    assert np.isclose(np.sum(w), 1.0), "Weights must sum to 1.0"
    assert w[0] > w[2], "Strongest model (m1) should get higher weight than weakest (m3)!"
    print("[+] Nelder-Mead OOF Weight Optimizer unit test passed perfectly!")
