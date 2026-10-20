import numpy as np
import pandas as pd

labels = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'
]
n = 10
test_ids = [f'study_{i}' for i in range(n)]

np.random.seed(42)
ours = pd.DataFrame(np.random.uniform(0.1, 0.9, (n, len(labels))), index=test_ids, columns=labels)
raw = pd.DataFrame(np.random.uniform(0.1, 0.9, (n, len(labels))), index=test_ids, columns=labels)
raw224 = pd.DataFrame(np.random.uniform(0.1, 0.9, (n, len(labels))), index=test_ids, columns=labels)
rad = pd.DataFrame(np.random.uniform(0.1, 0.9, (n, len(labels))), index=test_ids, columns=labels)

def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values

def pct_on(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any():
        r[ok] = pct(np.asarray(v, dtype=np.float64)[ok])
    return r

modes = [
    ('V32_FULL', {'Lateral Meniscus': 0.35, 'Synovitis': 0.30, 'ACL': 0.20, 'Contusion': 0.30}),
    ('V32_ABLATION', {c: 0.0 for c in labels})
]

for mode, W_RAD in modes:
    out = np.zeros((n, len(labels)))
    for j, c in enumerate(labels):
        r384 = pct(raw[c].values)
        r224 = pct(raw224[c].values)
        rr = 0.65 * r384 + 0.35 * r224
        ro = pct(ours[c].values)
        wr = 0.50
        primary = wr * rr + (1.0 - wr) * ro
        w_rad = W_RAD.get(c, 0.0)
        if w_rad > 0.0:
            r_rad = pct(rad[c].values)
            final_col = (1.0 - w_rad) * primary + w_rad * r_rad
        else:
            final_col = primary
        out[:, j] = final_col
    sub = pd.DataFrame(out.astype(np.float32), columns=labels)
    sub.insert(0, 'StudyInstanceUID', test_ids)
    assert not sub.isna().any().any(), f'{mode} has NaNs!'
    assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all(), f'{mode} out of bounds!'
    print(f'{mode}: Synthetic dry-run successful. Output shape: {sub.shape}')

print("ALL DRY-RUN TESTS PASSED PERFECTLY.")
