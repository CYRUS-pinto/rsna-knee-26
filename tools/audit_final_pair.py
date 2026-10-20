import pandas as pd
import numpy as np

f17 = "submissions/v17_multiscale_r6_0958/output/submission.csv"
f6 = "submissions/v06_pb_baseline/output/submission.csv"

df17 = pd.read_csv(f17).set_index("StudyInstanceUID")
df6 = pd.read_csv(f6).set_index("StudyInstanceUID")

print("V17 shape:", df17.shape)
print("V6 shape:", df6.shape)

common = df17.index.intersection(df6.index)
print(f"Common test studies in mock output: {len(common)}")

print("\n" + "="*70)
print(f"{'Pathology':<20} | {'V17 Mean':<10} | {'V6 Mean':<10} | {'Pearson r':<10} | {'Spearman rho':<10}")
print("="*70)

corrs = []
for col in df17.columns:
    v17 = df17.loc[common, col].values
    v6 = df6.loc[common, col].values
    
    # In case mock output has only 3 studies, calculate basic stats
    p_r = np.corrcoef(v17, v6)[0, 1] if len(common) > 1 and np.std(v17) > 0 and np.std(v6) > 0 else np.nan
    s17 = pd.Series(v17).rank()
    s6 = pd.Series(v6).rank()
    s_r = np.corrcoef(s17, s6)[0, 1] if len(common) > 1 and np.std(s17) > 0 and np.std(s6) > 0 else np.nan
    
    corrs.append((col, np.mean(v17), np.mean(v6), p_r, s_r))
    print(f"{col:<20} | {np.mean(v17):<10.4f} | {np.mean(v6):<10.4f} | {p_r:<10.4f} | {s_r:<10.4f}")

print("="*70)
