"""
5-Fold SWA Checkpoint Aggregator and Out-Of-Fold (OOF) Evaluator for Engine 6.
Enables:
1. Mathematical Stochastic Weight Averaging (SWA) across fold checkpoints to create a single zero-latency master checkpoint.
2. Complete 5-fold out-of-fold concatenation and official Wilcoxon-Mann-Whitney ROC-AUC validation.
3. Prepares Candidate V23 multi-fold SWA master submission.
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch

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

def compute_swa_checkpoint(checkpoint_paths, out_path):
    """
    Computes exact parameter-wise arithmetic mean across fold checkpoints.
    Zero-overhead ensembling: produces a single model weight file that captures multi-fold regularization.
    """
    assert len(checkpoint_paths) > 0, "No checkpoints provided for SWA"
    print(f"--- Computing SWA across {len(checkpoint_paths)} checkpoints ---")
    for p in checkpoint_paths:
        print(f"  + {p}")
        
    first_state = torch.load(checkpoint_paths[0], map_location='cpu')
    swa_state = {k: v.clone().float() for k, v in first_state.items()}
    
    for p in checkpoint_paths[1:]:
        st = torch.load(p, map_location='cpu')
        for k in swa_state:
            swa_state[k] += st[k].float()
            
    # Divide by count
    n = float(len(checkpoint_paths))
    for k in swa_state:
        swa_state[k] = (swa_state[k] / n).to(first_state[k].dtype)
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(swa_state, out_path)
    print(f"--> Saved SWA master checkpoint to {out_path} ({os.path.getsize(out_path) / (1024*1024):.1f} MB)")
    return out_path

def evaluate_oof_directory(oof_dir, ground_truth_path):
    """
    Concatenates fold validation prediction CSVs and calculates official competition metric.
    """
    df_gt = pd.read_parquet(ground_truth_path) if ground_truth_path.endswith('.parquet') else pd.read_csv(ground_truth_path)
    gt_dict = {row['StudyInstanceUID']: row[ALL_LABELS].values.astype(np.float32) for _, row in df_gt.iterrows()}
    
    oof_files = sorted(glob.glob(os.path.join(oof_dir, "*oof*.csv")))
    if not oof_files:
        print(f"No OOF files found in {oof_dir}")
        return None
        
    dfs = [pd.read_csv(f) for f in oof_files]
    merged = pd.concat(dfs, axis=0).drop_duplicates(subset=['StudyInstanceUID']).reset_index(drop=True)
    
    uids = merged['StudyInstanceUID'].tolist()
    y_pred = merged[ALL_LABELS].values
    y_true = np.stack([gt_dict[u] for u in uids], axis=0)
    
    aucs = {}
    for i, col in enumerate(ALL_LABELS):
        aucs[col] = exact_roc_auc(y_true[:, i], y_pred[:, i])
    macro = float(np.mean(list(aucs.values())))
    
    print("=" * 80)
    print(f"5-FOLD MERGED OUT-OF-FOLD (OOF) VALIDATION PERFORMANCE ({len(merged)} studies)")
    print("=" * 80)
    for col in ALL_LABELS:
        print(f"{col:<18}: {aucs[col]:.4f}")
    print("-" * 80)
    print(f"OFFICIAL MACRO ROC-AUC: {macro:.4f}")
    print("=" * 80)
    return macro, aucs

if __name__ == '__main__':
    print("5-Fold SWA Aggregator and OOF Evaluator initialized.")
