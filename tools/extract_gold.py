import pandas as pd
TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]
tr = pd.read_csv('train.csv')
gold = tr[tr[TARGETS].notna().all(axis=1)][['StudyInstanceUID'] + TARGETS]
print('Gold rows:', len(gold))
gold.to_csv('artifacts/gold_58_labels.csv', index=False)
csv_str = gold.to_csv(index=False)
print('CSV Length:', len(csv_str))
with open('artifacts/gold_58_raw.txt', 'w') as f:
    f.write(csv_str)
