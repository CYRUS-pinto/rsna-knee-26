import pandas as pd
import numpy as np

raw = pd.read_csv('scratch/quint_v6_out/raptor_raw.csv').set_index('StudyInstanceUID')
ours = pd.read_csv('scratch/quint_v6_out/ours_sub.csv').set_index('StudyInstanceUID')
sub_v6 = pd.read_csv('scratch/quint_v6_out/submission.csv').set_index('StudyInstanceUID')

labels = list(raw.columns)
n = len(raw)

def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values

W_RAPTOR = 0.50
W_RAPTOR_C = {
    'ACL': 0.65,
    'MCL': 0.60,
    'Contusion': 0.60,
    'Fracture': 0.55,
    'Medial OA': 0.30,
    'Lateral OA': 0.30,
    'PF OA': 0.30,
    'Effusion': 0.35,
    "Baker's": 0.35,
    'Medial Meniscus': 0.35,
    'Lateral Meniscus': 0.35,
    'Synovitis': 0.40
}

out = np.zeros((n, len(labels)))
for j, c in enumerate(labels):
    rr = pct(raw[c].values)
    ro = pct(ours[c].values)
    wr = W_RAPTOR_C.get(c, W_RAPTOR)
    out[:, j] = wr * rr + (1.0 - wr) * ro

sub_v7 = pd.DataFrame(out.astype(np.float32), columns=labels, index=raw.index)

print('=== V6 (Uniform 50/50) vs V7 (Asymmetric Specialization) ===')
for idx, sid in enumerate(raw.index):
    print(f"\n--- Study {idx} ---")
    comp = pd.DataFrame({
        'V6 (Uniform)': sub_v6.loc[sid],
        'V7 (Asymmetric)': sub_v7.loc[sid],
        'Shift (V7 - V6)': sub_v7.loc[sid] - sub_v6.loc[sid]
    })
    print(comp.round(4).to_string())
