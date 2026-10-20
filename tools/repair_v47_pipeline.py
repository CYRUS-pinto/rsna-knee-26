"""
Repair and Harden Candidate V47: Apex Tri-Arm Sovereign Ensemble.
1. Restores Cell 7 to the verified OrthoFoundation R6 controller runner (creates ours_sub.csv).
2. Sets Cell 8 to Fail-Soft Stacking with constant-prediction rejection (if e12/e13 are absent or degenerate, falls back 100% to 0.958 anchor).
3. Adds cyruspinto/rsna-knee-engine12-convnext-large-weights and cyruspinto/rsna-knee-engine13-swin-base-weights to kernel-metadata.json.
"""
import json
from pathlib import Path

v45_path = Path("submissions/v45_sovereign_rank_master/rsna-knee-v45-sovereign-rank-master.ipynb")
v47_path = Path("submissions/v47_apex_tri_arm_ensemble/rsna-knee-v47-apex-tri-arm-ensemble.ipynb")
meta_path = Path("submissions/v47_apex_tri_arm_ensemble/kernel-metadata.json")

v45_nb = json.load(open(v45_path, encoding="utf-8"))
v47_nb = json.load(open(v47_path, encoding="utf-8"))

# Cell 7 in V45 is the working probe_dual runner
v45_cell7 = v45_nb["cells"][7]

# Restore Cell 7 in V47 to probe_dual runner
v47_nb["cells"][7] = v45_cell7

stage3_code = '''# Stage 3: Candidate V47 Apex Tri-Arm Percentile Rank Stacking Engine
import os, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd

print("=== Starting Candidate V47 Apex Tri-Arm Percentile Rank Stacking ===", flush=True)

DATA = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
if not (DATA / 'test.csv').exists():
    DATA = Path('/kaggle/input/rsna-knee-abnormality-detection')
WORK = Path('/kaggle/working')

test_csv_path = DATA / 'test.csv'
test_df = pd.read_csv(test_csv_path, dtype={'StudyInstanceUID': str})
test_ids = test_df['StudyInstanceUID'].tolist()
N_STUDIES = len(test_ids)
print(f"Loaded {N_STUDIES} test studies.", flush=True)

ss_path = DATA / 'sample_submission.csv'
sample_sub = pd.read_csv(ss_path, dtype={'StudyInstanceUID': str}) if ss_path.exists() else None
labels = [c for c in sample_sub.columns if c != 'StudyInstanceUID'] if sample_sub is not None else [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
    'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# 1. Load Base 0.958 Raptor (384 + 224) + ConvNeXt R6 Anchor
ours = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(test_ids)
rsub = pd.read_csv(WORK / 'raptor_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(test_ids)
rsub224 = pd.read_csv(WORK / 'raptor224_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(test_ids)

# Calibrated Specialist Weights across Findings (V17 Benchmark)
W_RAPTOR_C = {
    'Medial OA': 0.45, 'Lateral OA': 0.20, 'PF OA': 0.55, 'Synovitis': 0.75,
    "Baker's": 0.30, 'ACL': 0.50, 'MCL': 0.45, 'Medial Meniscus': 0.45,
    'Lateral Meniscus': 0.45, 'Effusion': 0.45, 'Contusion': 0.50, 'Fracture': 0.50
}
A224 = 0.35

def pct_rank(s):
    arr = s.fillna(s.median() if not pd.isna(s.median()) else 0.5).values
    if len(arr) <= 1 or np.all(arr == arr[0]):
        return np.full_like(arr, 0.5, dtype=np.float32)
    return pd.Series(arr).rank(method='average', pct=True).values

anchor_preds = {}
for lbl in labels:
    r_val = (1.0 - A224) * rsub[lbl].values + A224 * rsub224[lbl].values
    o_val = ours[lbl].values
    wr = W_RAPTOR_C.get(lbl, 0.50)
    r_rk = pct_rank(pd.Series(r_val))
    o_rk = pct_rank(pd.Series(o_val))
    anchor_preds[lbl] = wr * r_rk + (1.0 - wr) * o_rk

print("[+] Anchor 0.958 baseline predictions generated successfully.", flush=True)

# 2. Check for auxiliary models (Engine 12 / Engine 13 / Engine 11)
e12_path = WORK / 'engine12_sub.csv'
e13_path = WORK / 'engine13_sub.csv'
e12_df = pd.read_csv(e12_path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(test_ids) if e12_path.exists() else None
e13_df = pd.read_csv(e13_path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(test_ids) if e13_path.exists() else None

final_dict = {'StudyInstanceUID': test_ids}

for lbl in labels:
    r_anchor = pct_rank(pd.Series(anchor_preds[lbl]))
    
    w_e12 = 0.0
    w_e13 = 0.0
    r_e12 = None
    r_e13 = None
    
    # Check if e12 predictions are valid (not constant dummy)
    if e12_df is not None and lbl in e12_df.columns:
        vals = e12_df[lbl].values
        if len(vals) > 1 and not np.all(vals == vals[0]):
            r_e12 = pct_rank(e12_df[lbl])
            w_e12 = 0.15
            
    # Check if e13 predictions are valid
    if e13_df is not None and lbl in e13_df.columns:
        vals = e13_df[lbl].values
        if len(vals) > 1 and not np.all(vals == vals[0]):
            r_e13 = pct_rank(e13_df[lbl])
            w_e13 = 0.15
            
    w_anchor = 1.0 - w_e12 - w_e13
    combined_rank = w_anchor * r_anchor
    if r_e12 is not None:
        combined_rank += w_e12 * r_e12
    if r_e13 is not None:
        combined_rank += w_e13 * r_e13
        
    final_dict[lbl] = np.clip(combined_rank, 0.0001, 0.9999)

sub_df = pd.DataFrame(final_dict)
sub_df.fillna(0.5, inplace=True)
for lbl in labels:
    sub_df[lbl] = sub_df[lbl].clip(0.0001, 0.9999)

sub_df.to_csv('submission.csv', index=False)
print("=== Successfully Generated submission.csv ===", flush=True)
print(f"Shape: {sub_df.shape}", flush=True)
print(sub_df.head(3), flush=True)
'''

v47_nb["cells"][8]["source"] = stage3_code.splitlines(keepends=True)

with open(v47_path, "w", encoding="utf-8") as f:
    json.dump(v47_nb, f, indent=1)

print("[+] Successfully patched V47 notebook: Cell 7 restored to probe_dual runner, Cell 8 fail-soft stacking!")

# Update kernel-metadata.json
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

sources = meta.get("dataset_sources", [])
required_sources = [
    "heliosli/rsna-knee-r6-frozen-fe055406-w1",
    "nartaa/rsna-knee-publication-swa-weights-20261007",
    "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
    "cyruspinto/rsna-knee-engine11-weights",
    "cyruspinto/rsna-knee-engine10-swin-weights",
    "cyruspinto/rsna-knee-engine12-convnext-large-weights",
    "cyruspinto/rsna-knee-engine13-swin-base-weights"
]
meta["dataset_sources"] = list(dict.fromkeys(sources + required_sources))

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("[+] Successfully updated kernel-metadata.json with Engine 12 & 13 datasets!")
