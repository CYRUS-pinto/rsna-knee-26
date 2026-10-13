import pandas as pd
import numpy as np
from scipy.special import expit, logit

train = pd.read_csv('train.csv')
FINDINGS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]
gold = train[train['ACL'].notna()].copy()
preds = pd.read_csv('SUPER_CONSENSUS_labels.csv').set_index('StudyInstanceUID')

def compute_auc(y_true, y_score):
    y_true = np.asarray(y_true, bool)
    y_score = np.asarray(y_score, float)
    pos = y_score[y_true]
    neg = y_score[~y_true]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    r = pd.Series(np.concatenate([pos, neg])).rank(method='average').to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

# Baseline Macro AUC
base_aucs = {}
for f in FINDINGS:
    y = gold[f].astype(int).values
    p = np.array([preds.loc[u, f] for u in gold['StudyInstanceUID']])
    base_aucs[f] = compute_auc(y, p)

print(f"Base Macro AUC: {np.mean(list(base_aucs.values())):.4f}")

# Cross-condition interaction test:
# Can Synovitis be boosted by Effusion evidence?
y_syn = gold['Synovitis'].astype(int).values
p_syn = np.array([preds.loc[u, 'Synovitis'] for u in gold['StudyInstanceUID']])
p_eff = np.array([preds.loc[u, 'Effusion'] for u in gold['StudyInstanceUID']])

for w_eff in [0.0, 0.05, 0.10, 0.15, 0.20]:
    p_syn_boosted = (1 - w_eff) * p_syn + w_eff * p_eff
    auc_boosted = compute_auc(y_syn, p_syn_boosted)
    print(f"Synovitis AUC with Effusion weight {w_eff:.2f}: {auc_boosted:.4f} (base: {base_aucs['Synovitis']:.4f})")

# Lateral OA boosted by Medial OA or PF OA?
y_loa = gold['Lateral OA'].astype(int).values
p_loa = np.array([preds.loc[u, 'Lateral OA'] for u in gold['StudyInstanceUID']])
p_moa = np.array([preds.loc[u, 'Medial OA'] for u in gold['StudyInstanceUID']])
p_pfoa = np.array([preds.loc[u, 'PF OA'] for u in gold['StudyInstanceUID']])

for w_oa in [0.0, 0.05, 0.10, 0.15]:
    p_loa_boosted = (1 - w_oa) * p_loa + w_oa * p_moa
    auc_boosted = compute_auc(y_loa, p_loa_boosted)
    print(f"Lateral OA AUC with Medial OA weight {w_oa:.2f}: {auc_boosted:.4f} (base: {base_aucs['Lateral OA']:.4f})")
