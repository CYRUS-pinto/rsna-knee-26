"""
Builder for Candidate V47: Apex Tri-Arm Sovereign Ensemble.
Combines:
1. Base Anchor (Weight 0.70): Proven 0.958 MultiScale Raptor (384+224) + ConvNeXt R6 Flip TTA.
2. Arm 2 (Weight 0.15): Engine 12 ConvNeXt-Large-384 Gated Attention MIL (structural OA, fractures).
3. Arm 3 (Weight 0.15): Engine 13 Swin-Base-384 Asymmetric Focal Loss (fine ligaments & menisci).
4. Rank Engine: Strict Percentile Rank Stacking (pct_on).
5. Guardrail Compliant: Zero fatal assertions, fail-soft fallbacks, [0.0001, 0.9999] clamping.
"""

import json
import os
from pathlib import Path

SRC_NOTEBOOK = Path("submissions/v45_sovereign_rank_master/rsna-knee-v45-sovereign-rank-master.ipynb")
DST_DIR = Path("submissions/v47_apex_tri_arm_ensemble")
DST_NOTEBOOK = DST_DIR / "rsna-knee-v47-apex-tri-arm-ensemble.ipynb"
DST_META = DST_DIR / "kernel-metadata.json"

assert SRC_NOTEBOOK.is_file(), f"Source notebook {SRC_NOTEBOOK} not found!"
DST_DIR.mkdir(parents=True, exist_ok=True)

nb = json.load(open(SRC_NOTEBOOK, "r", encoding="utf-8"))

# 1. Update Title Cell (Cell 0)
nb["cells"][0]["source"] = [
    "# RSNA Knee Candidate V47: Apex Tri-Arm Sovereign Ensemble\n",
    "\n",
    "**Championship Architecture (Breaking the 0.958 Plateau towards 0.967+):**\n",
    "1. **Anchor Backbone (0.70 Weight):** Raptor-384 + Raptor-224 + ConvNeXt R6 Flip TTA (Proven 0.958 PB Baseline).\n",
    "2. **Arm 2 - Depthwise Convolutional Specialist (0.15 Weight):** Engine 12 ConvNeXt-Large-384 Gated Attention MIL (OA, cartilage, trabecular impaction).\n",
    "3. **Arm 3 - Shifted-Window Attention Specialist (0.15 Weight):** Engine 13 Swin-Base-384 Asymmetric Focal MIL (cruciate & collateral ligaments, meniscal tears).\n",
    "4. **Percentile Rank Standardization (`pct_on`):** Strict ordinal rank fusion across orthogonal model families.\n",
    "5. **Fail-Soft Runtime Guarantee:** Full fallback to the 0.958 anchor if auxiliary weights are absent; zero fatal assertions; clamped to [0.0001, 0.9999].\n",
]

# 2. Update Blending Cell
cell8_code = """# Stage 3: Apex Tri-Arm Percentile Rank Stacking Engine
import csv, os, sys, time, json
import numpy as np
import pandas as pd

print("=== Starting Candidate V47 Apex Tri-Arm Percentile Rank Stacking ===", flush=True)

test_csv_path = DATA / 'test.csv'
test_df = pd.read_csv(test_csv_path, dtype={'StudyInstanceUID': str})
test_ids = test_df['StudyInstanceUID'].tolist()
N_STUDIES = len(test_ids)
print(f"Loaded {N_STUDIES} test studies.")

ss_path = DATA / 'sample_submission.csv'
sample_sub = pd.read_csv(ss_path, dtype={'StudyInstanceUID': str}) if ss_path.exists() else None
labels = [c for c in sample_sub.columns if c != 'StudyInstanceUID'] if sample_sub is not None else [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
    'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# 1. Load Base 0.958 Raptor + ConvNeXt R6 Anchor
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
    if len(arr) <= 1:
        return arr
    return pd.Series(arr).rank(method='average', pct=True).values

anchor_preds = {}
for lbl in labels:
    r_val = (1.0 - A224) * rsub[lbl].values + A224 * rsub224[lbl].values
    o_val = ours[lbl].values
    wr = W_RAPTOR_C.get(lbl, 0.50)
    # Calibrated anchor blend
    r_rk = pct_rank(pd.Series(r_val))
    o_rk = pct_rank(pd.Series(o_val))
    anchor_preds[lbl] = wr * r_rk + (1.0 - wr) * o_rk

print("Anchor predictions generated successfully.")

# 2. Check for Engine 12 (ConvNeXt-Large) & Engine 13 (Swin-Base) auxiliary predictions
e12_path = WORK / 'engine12_sub.csv'
e13_path = WORK / 'engine13_sub.csv'
e12_df = pd.read_csv(e12_path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(test_ids) if e12_path.exists() else None
e13_df = pd.read_csv(e13_path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(test_ids) if e13_path.exists() else None

final_dict = {'StudyInstanceUID': test_ids}

for lbl in labels:
    p_anchor = anchor_preds[lbl]
    r_anchor = pct_rank(pd.Series(p_anchor))
    
    w_anchor = 1.0
    w_e12 = 0.0
    w_e13 = 0.0
    
    r_e12 = None
    r_e13 = None
    
    if e12_df is not None and lbl in e12_df.columns:
        r_e12 = pct_rank(e12_df[lbl])
        w_e12 = 0.15
    if e13_df is not None and lbl in e13_df.columns:
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

# Format & Integrity checks
sub_df.fillna(0.5, inplace=True)
for lbl in labels:
    sub_df[lbl] = sub_df[lbl].clip(0.0001, 0.9999)

out_file = 'submission.csv'
sub_df.to_csv(out_file, index=False)
print(f"=== Successfully Generated {out_file} ===")
print(f"Output shape: {sub_df.shape}")
print(sub_df.head(3))
"""

# Find the blending cell in nb and replace it
replaced = False
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and any("W_RAPTOR" in line for line in cell["source"]):
        nb["cells"][idx]["source"] = cell8_code.splitlines(keepends=True)
        replaced = True
        print(f"[+] Replaced Blending Cell at Index {idx}")
        break

assert replaced, "Could not locate blending cell in source notebook!"

# Save Notebook
with open(DST_NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"[+] Successfully wrote {DST_NOTEBOOK}")

# Create Metadata JSON
meta = {
    "id": "cyruspinto/rsna-knee-v47-apex-tri-arm-ensemble",
    "title": "RSNA Knee V47 Apex Tri-Arm Ensemble",
    "code_file": "rsna-knee-v47-apex-tri-arm-ensemble.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [
        "heliosli/rsna-knee-r6-frozen-fe055406-w1",
        "nartaa/rsna-knee-publication-swa-weights-20261007",
        "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
        "cyruspinto/rsna-knee-engine11-weights",
        "cyruspinto/rsna-knee-engine10-swin-weights",
    ],
    "kernel_sources": [],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4",
}

with open(DST_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"[+] Successfully wrote {DST_META}")
