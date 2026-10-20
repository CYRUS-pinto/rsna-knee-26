import pandas as pd
import numpy as np

# Load competition ground truth, super consensus, and qwen soft labels
tf = pd.read_parquet('data/train_folds.parquet')
sc = pd.read_parquet('data/SUPER_CONSENSUS_labels.parquet')
qw = pd.read_csv('data/qwen_extracted_soft_labels.csv')

labels = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("=========================================================================================")
print("MEASURED AUDIT: SUPER_CONSENSUS vs QWEN-EXTRACTED vs train.csv (4,407 Studies)")
print("=========================================================================================")

print(f"{'Finding':<18} | {'train.csv Pos%':<14} | {'Qwen Pos%':<10} | {'Qwen Disagree%':<15} | {'SuperCons Diff':<15}")
print("-" * 80)

for l in labels:
    y_raw = tf[l].values
    y_sc = sc[l].values
    y_qw = qw[l].values
    
    pos_raw = (y_raw > 0.5).mean() * 100
    pos_qw = (y_qw > 0.5).mean() * 100
    
    # Binary disagreement of Qwen vs train.csv
    qwen_disagree = ((y_raw > 0.5) != (y_qw > 0.5)).mean() * 100
    
    # Continuous mean diff of SuperCons vs train.csv
    sc_mean_diff = np.mean(np.abs(y_raw - y_sc))
    
    print(f"{l:<18} | {pos_raw:6.2f}%       | {pos_qw:6.2f}%   | {qwen_disagree:6.2f}%         | {sc_mean_diff:8.4f}")
