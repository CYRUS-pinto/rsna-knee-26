import pandas as pd

df = pd.read_csv('train.csv')
TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]
cols = ['StudyInstanceUID'] + TARGETS
gold = df[df[TARGETS].notna().all(axis=1)][cols]
print(f"Total gold rows: {len(gold)}")

csv_str = gold.to_csv(index=False)
lines = [l.strip() for l in csv_str.strip().split('\n') if l.strip()]

# Test parsing with pandas
from io import StringIO
test_df = pd.read_csv(StringIO(csv_str))
print(f"Parsed shape: {test_df.shape}")
assert test_df.shape == (58, 13), f"Wrong shape: {test_df.shape}"
print("ASSERTION PASSED: Exact 58x13 DataFrame!")

with open('tools/gold_58_clean.csv', 'w', newline='') as f:
    f.write(csv_str)
print("Saved to tools/gold_58_clean.csv")
