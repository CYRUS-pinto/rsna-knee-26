import pandas as pd
import numpy as np

# Load ground truth from train.csv
train = pd.read_csv('train.csv')
FINDINGS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

gold = train[train['ACL'].notna()].copy()
print(f'Total Gold Adjudicated Studies: {len(gold)}')

# Load our SUPER_CONSENSUS_labels.csv
preds = pd.read_csv('SUPER_CONSENSUS_labels.csv').set_index('StudyInstanceUID')

# Compute per-finding AUC and Brier score
def compute_auc(y_true, y_score):
    y_true = np.asarray(y_true, bool)
    y_score = np.asarray(y_score, float)
    pos = y_score[y_true]
    neg = y_score[~y_true]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    r = pd.Series(np.concatenate([pos, neg])).rank(method='average').to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

results = []
all_y = []
all_p = []

for f in FINDINGS:
    y = gold[f].astype(int).values
    p = [preds.loc[u, f] for u in gold['StudyInstanceUID']]
    a = compute_auc(y, p)
    brier = np.mean((p - y) ** 2)
    pos_rate = np.mean(y)
    mean_pred = np.mean(p)
    results.append({
        'Finding': f,
        'ROC_AUC': a,
        'Brier_Loss': brier,
        'Gold_Pos_Rate': pos_rate,
        'Mean_Pred': mean_pred,
        'Bias': mean_pred - pos_rate
    })
    all_y.extend(y)
    all_p.extend(p)

res_df = pd.DataFrame(results)
print('\n=== PER-FINDING VALIDATION ON 58 GOLD STUDIES ===')
print(res_df.to_string(index=False, justify='left', float_format=lambda x: f'{x:.4f}'))
print(f'\nMACRO ROC AUC: {res_df["ROC_AUC"].mean():.4f}')
print(f'MEAN BRIER LOSS: {res_df["Brier_Loss"].mean():.4f}')

# Overall calibration across probability bins
all_y = np.array(all_y)
all_p = np.array(all_p)
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
cal_df = pd.DataFrame({'truth': all_y, 'prob': all_p})
cal_df['bin'] = pd.cut(cal_df['prob'], bins, right=False)
cal_summary = cal_df.groupby('bin', observed=True)['truth'].agg(['count', 'mean'])
cal_summary.rename(columns={'mean': 'observed_pos_rate'}, inplace=True)

print('\n=== OVERALL CALIBRATION CURVE ===')
print(cal_summary.to_string(float_format=lambda x: f'{x:.3f}'))
