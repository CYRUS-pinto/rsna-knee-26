"""
Pathology-Specific Probability Stacking Optimizer
Solves for the optimal ensemble weight matrix W (shape: [12, num_models])
to maximize ROC-AUC across each individual target condition independently.
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

def optimize_pathology_weights(model_oof_dict, ground_truth_df):
    """
    model_oof_dict: dict of {model_name: df_with_predictions}
    ground_truth_df: df with StudyInstanceUID and TARGETS
    """
    model_names = list(model_oof_dict.keys())
    M = len(model_names)
    weights_matrix = np.zeros((len(TARGETS), M))
    
    print(f"[*] Optimizing Pathology-Specific Stacking across {M} model streams...")
    
    for t_idx, target in enumerate(TARGETS):
        # Align targets
        y_true = ground_truth_df[target].values
        valid_mask = ~np.isnan(y_true)
        y_valid = y_true[valid_mask]
        
        preds_stack = np.column_stack([
            model_oof_dict[m][target].values[valid_mask] for m in model_names
        ])
        
        def objective(w):
            w_pos = np.maximum(0, w)
            sum_w = np.sum(w_pos)
            if sum_w == 0:
                w_norm = np.ones(M) / M
            else:
                w_norm = w_pos / sum_w
            blend = preds_stack @ w_norm
            try:
                score = roc_auc_score(y_valid, blend)
                return -score
            except Exception:
                return 0.0

        init_w = np.ones(M) / M
        res = minimize(objective, init_w, method='Nelder-Mead', options={'maxiter': 300})
        
        w_opt = np.maximum(0, res.x)
        w_opt = w_opt / (np.sum(w_opt) + 1e-8)
        weights_matrix[t_idx] = w_opt
        
        best_auc = -res.fun
        print(f"  Target: {target:<18} | Optimized AUC: {best_auc:.4f} | Weights: {np.round(w_opt, 3)}")

    return dict(zip(TARGETS, weights_matrix.tolist()))

if __name__ == '__main__':
    print("[+] Pathology-Specific Optimizer Module Loaded.")
