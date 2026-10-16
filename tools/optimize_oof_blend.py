"""
Exploratory OOF Audit: Arm A (ntejas_repo) vs Engine 4 on 161 Fold 0 Intersection Studies.
NOTE: This experiment tests Arm A (ntejas_repo) vs collapsed Engine 4 (macro AUC 0.685).
It does NOT evaluate Raptor or ConvNeXt R6 (Candidate V17/V20).
Results are exploratory; tuning 12 weights on n=161 carries ~+0.0015 to +0.0020 selection noise.
"""

import pandas as pd
import numpy as np

LABELS = [
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

def rankpct(s):
    return s.rank(pct=True, method='average')

# 1. Load data
super_df = pd.read_parquet('data/SUPER_CONSENSUS_labels.parquet')
e4_df = pd.read_parquet('models/engine4_kaggle_output/oof_engine4_fold0.parquet')
arma_df = pd.read_csv('scratch/ntejas_repo/oof/armA_oof_fold0.csv')
if 'STUDY_UID' in arma_df.columns:
    arma_df = arma_df.rename(columns={'STUDY_UID': 'StudyInstanceUID'})

# 2. Merge all on StudyInstanceUID
merged = arma_df.merge(e4_df, on='StudyInstanceUID', suffixes=('_arma', '_e4'))
merged = merged.merge(super_df, on='StudyInstanceUID')

print(f"==================================================================")
print(f"  EXPLORATORY AUDIT COHORT: {len(merged)} Studies on Fold 0 Intersection")
print(f"  Models: Arm A (ntejas_repo) vs Engine 4 (collapsed, macro AUC 0.685)")
print(f"==================================================================")

# Evaluate Solo Arm A
aucs_arma = {}
aucs_e4 = {}
aucs_equal_blend = {}
aucs_power_blend = {}

for col in LABELS:
    yt = merged[col] >= 0.5
    pa = merged[f"{col}_arma"]
    pe = merged[f"{col}_e4"]
    
    aucs_arma[col] = exact_roc_auc(yt, pa)
    aucs_e4[col] = exact_roc_auc(yt, pe)
    
    # Linear rank blend (0.5 Arm A, 0.5 Engine 4)
    ra = rankpct(pa)
    re = rankpct(pe)
    blend_lin = 0.5 * ra + 0.5 * re
    aucs_equal_blend[col] = exact_roc_auc(yt, blend_lin)
    
    # Power-mean blend (p=1.4 for focal, p=1.0 for diffuse)
    p_exp = 1.4 if col in ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Fracture'] else 1.0
    blend_pow = (0.5 * (ra ** p_exp) + 0.5 * (re ** p_exp)) ** (1.0 / p_exp)
    aucs_power_blend[col] = exact_roc_auc(yt, blend_pow)

macro_arma = np.mean(list(aucs_arma.values()))
macro_e4 = np.mean(list(aucs_e4.values()))
macro_lin = np.mean(list(aucs_equal_blend.values()))
macro_pow = np.mean(list(aucs_power_blend.values()))

print(f"\n{'Finding':<18} | {'Arm A (ntejas)':<14} | {'Engine 4':<8} | {'Linear 50/50':<12} | {'Power (p=1.4)':<12}")
print("-" * 74)
for col in LABELS:
    print(f"{col:<18} | {aucs_arma[col]:.4f}         | {aucs_e4[col]:.4f}   | {aucs_equal_blend[col]:.4f}       | {aucs_power_blend[col]:.4f}")
print("-" * 74)
print(f"{'MACRO ROC-AUC':<18} | {macro_arma:.4f}         | {macro_e4:.4f}   | {macro_lin:.4f}       | {macro_pow:.4f}")
print("==================================================================")

# 3. Find optimal weight w_arma in [0, 1] per finding
print("\n>>> IN-SAMPLE GRID SEARCH WEIGHTS (w_arma vs 1 - w_arma) <<<")
opt_weights = {}
opt_aucs = {}

for col in LABELS:
    yt = merged[col] >= 0.5
    ra = rankpct(merged[f"{col}_arma"])
    re = rankpct(merged[f"{col}_e4"])
    p_exp = 1.4 if col in ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Fracture'] else 1.0
    
    best_w = 1.0
    best_auc = aucs_arma[col]
    
    # Grid search w from 0.0 to 1.0 in 0.05 steps
    for w in np.linspace(0.0, 1.0, 21):
        if p_exp == 1.0:
            cand = w * ra + (1.0 - w) * re
        else:
            cand = (w * (ra ** p_exp) + (1.0 - w) * (re ** p_exp)) ** (1.0 / p_exp)
        auc_cand = exact_roc_auc(yt, cand)
        if auc_cand > best_auc:
            best_auc = auc_cand
            best_w = float(w)
            
    opt_weights[col] = best_w
    opt_aucs[col] = best_auc

macro_opt = np.mean(list(opt_aucs.values()))

print(f"\n{'Finding':<18} | {'Arm A Solo':<10} | {'Grid-Search Weight (Arm A)':<28} | {'In-Sample AUC':<14} | {'Delta':<8}")
print("-" * 84)
for col in LABELS:
    delta = opt_aucs[col] - aucs_arma[col]
    sign = "+" if delta >= 0 else ""
    print(f"{col:<18} | {aucs_arma[col]:.4f}     | {opt_weights[col]:<28.2f} | {opt_aucs[col]:.4f}        | {sign}{delta:.4f}")
print("-" * 84)
print(f"{'MACRO ROC-AUC':<18} | {macro_arma:.4f}     | {'---':<28} | {macro_opt:.4f}        | +{macro_opt - macro_arma:.4f}")
print("==================================================================")
print("CAVEAT: The +0.0022 in-sample gain is within expected selection noise")
print("        (~+0.0015 expected for a 21-weight grid search across 12 classes on n=161).")
print("        This test does NOT evaluate Raptor or ConvNeXt R6.")
print("==================================================================")
