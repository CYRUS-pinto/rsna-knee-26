import json
import os

src_nb = 'submissions/v24_r7_uniform_multiscale_master/rsna-knee-v24-r7-uniform-multiscale-master.ipynb'
dst_dir = 'submissions/v26_fast_raptor_multiscale'
os.makedirs(dst_dir, exist_ok=True)

with open(src_nb, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Header markdown
md_header = """# RSNA Knee - Candidate V26 Fast MultiScale Raptor Measurement
#
# PURPOSE: Frontier measurement on the $18,000 Efficiency Prize track.
# Evaluates standalone Multi-Scale Raptor (384px + 224px SWA) without ConvNeXt.
# Expected runtime on hidden test set: ~5 to 7 minutes (shaving 80% off dual-arm latency).
"""

# Extract stages
# Cell 0 is md
# Cell 1 is md
# Cell 2 is raptor 384 write
# Cell 3 is raptor 384 run
# Cell 4 is raptor 224 write
# Cell 5 is raptor 224 run
c0 = {"cell_type": "markdown", "metadata": {}, "source": [md_header]}
c2 = nb['cells'][2]  # raptor 384 writer
c3 = nb['cells'][3]  # raptor 384 run
c4 = nb['cells'][4]  # raptor 224 writer
c5 = nb['cells'][5]  # raptor 224 run

# Build fast blend cell
fast_blend_code = """# Stage 3: Multi-Scale Pure Raptor Rank Fusion
# Combines Raptor 384px and Raptor 224px via monotonic percentile rank averaging.
import csv
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path('/kaggle/input/rsna-knee-abnormality-detection')
if not DATA.exists():
    DATA = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
WORK = Path('/kaggle/working')

test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
ss = DATA / 'sample_submission.csv'
cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else None
labels = cols[1:] if cols else [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

raw384 = pd.read_csv(WORK / 'raptor_raw.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
raw224 = pd.read_csv(WORK / 'raptor224_raw.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]

n = len(test_ids)
r384_fail = (raw384.values == 0.5).all(axis=1)
r224_fail = (raw224.values == 0.5).all(axis=1)
rap_fail = r384_fail & r224_fail

if r384_fail.sum() > 0.05 * n:
    print(f"WARNING: Raptor-384 failed on {r384_fail.sum()} of {n} studies")
if r224_fail.sum() > 0.05 * n:
    print(f"WARNING: Raptor-224 failed on {r224_fail.sum()} of {n} studies")

def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values

def pct_on(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any():
        r[ok] = pct(np.asarray(v, dtype=np.float64)[ok])
    return r

A224 = 0.35  # Verified optimal multiscale split: 65% 384px, 35% 224px
out = np.zeros((n, len(labels)))

for j, c in enumerate(labels):
    r384 = pct_on(raw384[c].values, ~r384_fail)
    r224 = pct_on(raw224[c].values, ~r224_fail)
    rr = np.where(r384_fail, r224, np.where(r224_fail, r384, (1.0 - A224) * r384 + A224 * r224))
    rr[rap_fail] = 0.5
    out[:, j] = rr

sub = pd.DataFrame(out.astype(np.float32), columns=labels)
sub.insert(0, 'StudyInstanceUID', test_ids)
if cols:
    sub = sub[cols]

assert sub['StudyInstanceUID'].tolist() == test_ids
assert np.isfinite(sub[labels].values).all()
assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()

sub.to_csv(WORK / 'submission.csv', index=False)
receipt = {
    'architecture': 'Raptor-MultiScale-Fast (384 + 224)',
    'rows': n,
    'a224_weight': A224,
    'r384_failed': int(r384_fail.sum()),
    'r224_failed': int(r224_fail.sum()),
    'both_failed': int(rap_fail.sum())
}
(WORK / 'blend_receipt.json').write_text(json.dumps(receipt, indent=2))
print("FAST_RAPTOR_COMPLETE:", json.dumps(receipt))
display(sub.head())
"""

c_blend = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [fast_blend_code]}

fast_nb = {
    "metadata": nb['metadata'],
    "nbformat": nb['nbformat'],
    "nbformat_minor": nb['nbformat_minor'],
    "cells": [c0, c2, c3, c4, c5, c_blend]
}

dst_nb_path = os.path.join(dst_dir, 'rsna-knee-fast-raptor-multiscale.ipynb')
with open(dst_nb_path, 'w', encoding='utf-8') as f:
    json.dump(fast_nb, f, indent=1)

meta = {
  "id": "cyruspinto/rsna-knee-fast-raptor-multiscale",
  "title": "RSNA Knee Fast Raptor MultiScale Frontier Measurement",
  "code_file": "rsna-knee-fast-raptor-multiscale.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "keywords": ["gpu"],
  "dataset_sources": [
    "nartaa/rsna-knee-publication-swa-weights-20261007"
  ],
  "kernel_sources": [],
  "competition_sources": ["rsna-knee-abnormality-detection"],
  "model_sources": [],
  "machine_shape": "NvidiaTeslaT4"
}

with open(os.path.join(dst_dir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

print("[+] Successfully built Fast MultiScale Raptor notebook and metadata in", dst_dir)
