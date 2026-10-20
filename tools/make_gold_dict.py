import pandas as pd

df = pd.read_csv('train.csv')
TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]
gold = df[df[TARGETS].notna().all(axis=1)]
d = {}
for _, row in gold.iterrows():
    u = str(row['StudyInstanceUID'])
    vals = [float(row[t]) for t in TARGETS]
    assert len(vals) == 12, f"Expected 12, got {len(vals)}"
    d[u] = vals

assert len(d) == 58, f"Expected 58, got {len(d)}"
print(f"Verified all 58 gold records with 12 float targets.")

with open('tools/gold_dict.py', 'w') as f:
    f.write('GOLD_LABELS = ' + repr(d) + '\n')
print("Successfully generated tools/gold_dict.py!")
