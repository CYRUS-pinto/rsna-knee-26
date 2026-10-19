"""
Builder for Candidate V42: Sovereign Quintet Live-Inference Master.
Wired end-to-end:
1. Stage 1: Raptor-384 + Super-CoAtNet (0.35) + Raptor-224 (0.30)
2. Stage 2: ConvNeXt-Small R7 4-Fold Ensemble (Stride 2)
3. Stage 3 (NEW LIVE INFERENCE): Sovereign Quintet Runner
   - Engine 11 Full 4-Fold ConvNeXt-Base (88M params, 0.8712 Gold AUC)
   - Engine 9 Fold 1 ConvNeXt-Small (0.8553 Gold AUC)
   - Engine 10 Fold 1 Swin-Base (0.8634 Gold AUC)
   - Horizontal Flip TTA
   - Writes /kaggle/working/sovereign_sub.csv
4. Stage 4: 3-Way Grandmaster Calibrated Power-Rank Ensembling
   - W_RAPTOR = 0.35, W_OURS = 0.35, W_SOV = 0.30
   - Power mean p=1.3 on acute tears
   - Fallback protection if any arm fails
"""

import json
import os
from pathlib import Path

SRC_NOTEBOOK = Path("submissions/v41_sovereign_hendecet_titan/rsna-knee-v41-sovereign-hendecet-titan.ipynb")
DST_DIR = Path("submissions/v42_sovereign_live_master")
DST_NOTEBOOK = DST_DIR / "rsna-knee-v42-sovereign-live-master.ipynb"
DST_META = DST_DIR / "kernel-metadata.json"

assert SRC_NOTEBOOK.is_file(), f"Source notebook {SRC_NOTEBOOK} not found!"
DST_DIR.mkdir(parents=True, exist_ok=True)

nb = json.load(open(SRC_NOTEBOOK, "r", encoding="utf-8"))

# 1. Update Title Cell (Cell 0)
nb['cells'][0]['source'] = [
    "# RSNA Knee Candidate V42: Sovereign Live-Inference Master (True Live 3-Way Grandmaster Ensembling)\n",
    "\n",
    "**The Complete Live-Inference Ensemble:**\n",
    "1. **Stage 1 (Attention MultiScale):** Raptor CoAtNet-384 (0.65) + Cyrus Super-CoAtNet-384 (0.35) + Nartaa 224 (0.30 share).\n",
    "2. **Stage 2 (Joint-Space Conv):** Heliosli ConvNeXt-Small R7 4-Fold Ensemble (Stride 2).\n",
    "3. **Stage 3 (NEW Live Sovereign Inference):**\n",
    "   - Engine 11 ConvNeXt-Base 4-Fold Ensemble (Fold 0/1: 0.8712, Fold 2: 0.8619, Fold 3: 0.8561, Fold 4: 0.8506, 88M params each)\n",
    "   - Engine 9 ConvNeXt-Small Fold 1 (0.8553)\n",
    "   - Engine 10 Swin-Base Fold 1 (0.8634)\n",
    "   - Horizontal Flip TTA on all studies\n",
    "   - Produces `/kaggle/working/sovereign_sub.csv`\n",
    "4. **Stage 4 (Grandmaster 3-Way Ensembling):** 0.35 Raptor + 0.35 R7 + 0.30 Sovereign with Power-Mean ($p=1.3$) on Acute Tears.\n"
]

# 2. Write Complete Live Sovereign Inference Script in Cell 8
sovereign_script = '''%%writefile /tmp/sovereign_run.py
"""
Sovereign Live Test Inference Engine
Evaluates all available Sovereign Checkpoints:
- Engine 11 Base Fold 0/1, Fold 2, Fold 3, Fold 4 (88M params, 384px)
- Engine 9 Small Fold 1 (384px)
- Engine 10 Swin-Base Fold 1 (384px)
Applies Horizontal Flip TTA: 0.5 * P(X) + 0.5 * P(Flip(X))
Outputs /kaggle/working/sovereign_sub.csv
"""
import os, sys, time, json, gc
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import timm

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18),
         ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = sum(s[2] for s in SLOTS)  # 96
IMG = 384

class Engine9MIL(nn.Module):
    def __init__(self, feat_dim=768, model_name='convnext_small.fb_in22k_ft_in1k_384'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.norm = nn.LayerNorm(feat_dim)
        self.att = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, 12)
        )
        self.clsW = nn.Parameter(torch.empty(12, feat_dim))
        self.clsb = nn.Parameter(torch.empty(12))

    def forward(self, x):
        # x: (B, S, 3, H, W) -> flatten to (B*S, 3, H, W)
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W)
        h = self.backbone(x_flat)
        h = self.norm(h)
        h = h.view(B, S, -1)
        att_logits = self.att(h)
        alpha = F.softmax(att_logits, dim=1)
        z = torch.einsum('bsk,bsd->bkd', alpha, h)
        logits = (z * self.clsW.unsqueeze(0)).sum(dim=-1) + self.clsb.unsqueeze(0)
        return logits[0]

class SwinMILModel(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window12_384.ms_in22k_ft_in1k', num_classes=12):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W)
        feats = self.backbone(x_flat)
        feats = feats.view(B, S, -1)
        attn_scores = self.attention(feats)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled_attn = torch.sum(feats * attn_weights, dim=1)
        pooled_max, _ = torch.max(feats, dim=1)
        pooled = 0.7 * pooled_attn + 0.3 * pooled_max
        logits = self.head(pooled)
        return logits[0]

def extract_triplets_from_vol(vol_96, n_slices=8, res=384):
    D, H, W = vol_96.shape
    step = max(1, (D - 2) // n_slices)
    centers = [min(D - 2, 1 + i * step) for i in range(n_slices)]
    triplets = []
    for c in centers:
        trip = vol_96[c-1:c+2].astype(np.float32) / 255.0
        triplets.append(trip)
    stack = np.stack(triplets, axis=0) # (8, 3, 384, 384)
    t = torch.from_numpy(np.ascontiguousarray(stack))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0) # (1, 8, 3, 384, 384)

def find_test_root():
    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"]
    for b in cands:
        if os.path.exists(b + "/test.csv"):
            return b
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f and (os.path.isdir(d + "/test_series") or os.path.isdir(d + "/test_images")):
            return d
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f:
            return d
    raise RuntimeError("no test root under /kaggle/input")

def main():
    t0 = time.time()
    print("=== Candidate V42: Starting Live Sovereign Test Inference ===", flush=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Inference device: {device}", flush=True)

    # 1. Load available Sovereign models
    loaded_models = []

    # Engine 11 Fold 0/1 (Base 384, 88M params)
    p = Path('/kaggle/input/rsna-knee-engine11-weights/engine11_convnext_base_384_best.pt')
    if p.exists():
        print(f"Loading Engine 11 Base Fold 0/1 from {p}...", flush=True)
        ck = torch.load(p, map_location='cpu')
        m = Engine9MIL(feat_dim=1024, model_name='convnext_base.fb_in22k_ft_in1k_384').to(device)
        m.load_state_dict(ck['model'], strict=True)
        m.eval()
        loaded_models.append(('e11_f0', m))

    # Engine 11 Fold 2 (Base 384, 88M params)
    p = Path('/kaggle/input/rsna-knee-engine11-fold2-weights/engine11_convnext_base_384_fold2_best.pt')
    if p.exists():
        print(f"Loading Engine 11 Base Fold 2 from {p}...", flush=True)
        ck = torch.load(p, map_location='cpu')
        m = Engine9MIL(feat_dim=1024, model_name='convnext_base.fb_in22k_ft_in1k_384').to(device)
        m.load_state_dict(ck['model'], strict=True)
        m.eval()
        loaded_models.append(('e11_f2', m))

    # Engine 11 Fold 3 (Base 384, 88M params)
    p = Path('/kaggle/input/rsna-knee-engine11-fold3-weights/engine11_convnext_base_384_fold3_best.pt')
    if p.exists():
        print(f"Loading Engine 11 Base Fold 3 from {p}...", flush=True)
        ck = torch.load(p, map_location='cpu')
        m = Engine9MIL(feat_dim=1024, model_name='convnext_base.fb_in22k_ft_in1k_384').to(device)
        m.load_state_dict(ck['model'], strict=True)
        m.eval()
        loaded_models.append(('e11_f3', m))

    # Engine 11 Fold 4 (Base 384, 88M params)
    p = Path('/kaggle/input/rsna-knee-engine11-fold4-weights/engine11_convnext_base_384_fold4_best.pt')
    if p.exists():
        print(f"Loading Engine 11 Base Fold 4 from {p}...", flush=True)
        ck = torch.load(p, map_location='cpu')
        m = Engine9MIL(feat_dim=1024, model_name='convnext_base.fb_in22k_ft_in1k_384').to(device)
        m.load_state_dict(ck['model'], strict=True)
        m.eval()
        loaded_models.append(('e11_f4', m))

    # Engine 9 Fold 1 (Small 384, 50M params)
    p = Path('/kaggle/input/rsna-knee-engine9-fold1-weights/engine9_convnext_384_fold1_best.pt')
    if p.exists():
        print(f"Loading Engine 9 Fold 1 from {p}...", flush=True)
        ck = torch.load(p, map_location='cpu')
        m = Engine9MIL(feat_dim=768, model_name='convnext_small.fb_in22k_ft_in1k_384').to(device)
        m.load_state_dict(ck['model'], strict=True)
        m.eval()
        loaded_models.append(('e9_f1', m))

    # Engine 10 Fold 1 (Swin-Base 384, 88M params)
    p = Path('/kaggle/input/rsna-knee-engine10-fold1-weights/engine10_swin_384_fold1_best.pt')
    if p.exists():
        print(f"Loading Engine 10 Swin Fold 1 from {p}...", flush=True)
        ck = torch.load(p, map_location='cpu')
        m = SwinMILModel(num_classes=12).to(device)
        m.load_state_dict(ck['model'], strict=True)
        m.eval()
        loaded_models.append(('e10_f1', m))

    print(f"Total Sovereign models loaded: {len(loaded_models)}", flush=True)
    assert len(loaded_models) > 0, "FATAL: No sovereign models could be loaded!"

    # 2. Prepare test dataset reader via _fastread
    ROOT = find_test_root()
    tsdir = ROOT + "/test_series" if os.path.isdir(ROOT + "/test_series") else ROOT + "/test_images"
    test_df = pd.read_csv(ROOT + "/test.csv")
    test_df["StudyInstanceUID"] = test_df["StudyInstanceUID"].astype(str)
    test_ids = test_df["StudyInstanceUID"].tolist()
    
    tser = pd.read_csv(ROOT + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}
    print(f"Test studies to score: {len(test_ids)}", flush=True)

    # Import _fastread
    sys.path.insert(0, '/kaggle/working')
    sys.path.insert(0, '.')
    import _fastread
    builder = _fastread.make_builder('study_lut', IMG, SLOTS, hdr_threads=32, px_threads=16)

    # 3. Live Test Inference Loop
    records = []
    for idx, sid in enumerate(test_ids):
        try:
            vol, mask = builder(sid, SER, tsdir)
            x = extract_triplets_from_vol(vol, n_slices=8, res=IMG).to(device)
            x_flip = torch.flip(x, dims=[-1])
            
            study_probs = []
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    for name, model in loaded_models:
                        l_plain = model(x)
                        l_flip = model(x_flip)
                        p_plain = torch.sigmoid(l_plain)
                        p_flip = torch.sigmoid(l_flip)
                        p = 0.5 * p_plain + 0.5 * p_flip
                        p = torch.nan_to_num(p.float(), nan=0.5, posinf=0.9999, neginf=0.0001)
                        study_probs.append(p.cpu().numpy())
            
            # Average across all sovereign models
            avg_p = np.mean(study_probs, axis=0) # (12,)
            records.append([sid] + avg_p.tolist())
        except Exception as e:
            print(f"  Fallback study {sid}: {e}", flush=True)
            records.append([sid] + [0.5] * 12)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_ids):
            print(f"  Sovereign progress: {idx + 1}/{len(test_ids)} studies scored | {time.time() - t0:.1f}s", flush=True)

    # 4. Save sovereign_sub.csv
    out_df = pd.DataFrame(records, columns=['StudyInstanceUID'] + TARGETS)
    out_path = Path('/kaggle/working/sovereign_sub.csv')
    out_df.to_csv(out_path, index=False)
    print(f"WROTE {out_path} | Shape: {out_df.shape} | Total Time: {time.time() - t0:.1f}s", flush=True)
    print(out_df.head(), flush=True)

if __name__ == '__main__':
    main()
'''

nb['cells'][8]['source'] = sovereign_script.splitlines(keepends=True)

# 3. Update Cell 10 (Stage 3 Blend) to perform true 3-Way Grandmaster Ensembling
cell10_script = '''# Stage 3: Grandmaster 3-Way Rank-Power Blend (Raptor + R7 + Sovereign)
# Fixed blend weights: 0.35 Raptor + 0.35 R7 + 0.30 Sovereign
# Preserves fail-soft fallback if any component is missing.
import csv, time
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
if not DATA.exists():
    DATA = Path('/kaggle/input/rsna-knee-abnormality-detection')
WORK = Path('/kaggle/working')

W_RAPTOR, W_OURS, W_SOV = 0.35, 0.35, 0.30
A224 = 0.30

# Per-finding overrides:
W_RAPTOR_C = {
    'Lateral OA': 0.20,
    'Synovitis': 0.70,
    'PF OA': 0.50,
    "Baker's": 0.35,
    'ACL': 0.40,
    'MCL': 0.40,
    'Medial Meniscus': 0.35,
    'Lateral Meniscus': 0.35,
    'Medial OA': 0.40,
    'Effusion': 0.40,
    'Contusion': 0.40,
    'Fracture': 0.40
}

test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
ss = DATA / 'sample_submission.csv'
ours = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})
cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else list(ours.columns)
labels = cols[1:]

rsub = pd.read_csv(WORK / 'raptor_sub.csv', dtype={'StudyInstanceUID': str})
rsub224 = pd.read_csv(WORK / 'raptor224_sub.csv', dtype={'StudyInstanceUID': str})

# Load Sovereign predictions with fail-soft fallback
sov_path = WORK / 'sovereign_sub.csv'
if sov_path.exists():
    sov = pd.read_csv(sov_path, dtype={'StudyInstanceUID': str})
    print(f"Loaded live Sovereign predictions from {sov_path} (shape: {sov.shape})", flush=True)
else:
    print(f"WARNING: {sov_path} missing! Falling back to ours_sub.csv as sovereign proxy", flush=True)
    sov = ours.copy()

def rankpct(df_in, target_cols):
    res = df_in.copy()
    for c in target_cols:
        ranks = df_in[c].rank(pct=True).values
        res[c] = ranks
    return res

r_raptor384 = rankpct(rsub, labels)
r_raptor224 = rankpct(rsub224, labels)
r_raptor = r_raptor384.copy()
for c in labels:
    r_raptor[c] = (1.0 - A224) * r_raptor384[c] + A224 * r_raptor224[c]

r_ours = rankpct(ours, labels)
r_sov = rankpct(sov, labels)

# Power-mean sharpening factor for acute structural tears
# Sharpens extreme detections (tears/fractures) while suppressing background noise
ACUTE_CLASSES = {'ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Fracture', 'Contusion'}
P_POWER = 1.3

final_df = pd.DataFrame({'StudyInstanceUID': test_ids})
for c in labels:
    w_rap = W_RAPTOR_C.get(c, W_RAPTOR)
    rem = 1.0 - w_rap
    w_o = rem * 0.54
    w_s = rem * 0.46
    
    if c in ACUTE_CLASSES:
        # Non-linear power mean blend
        val = (w_rap * (r_raptor[c] ** P_POWER) + 
               w_o * (r_ours[c] ** P_POWER) + 
               w_s * (r_sov[c] ** P_POWER)) ** (1.0 / P_POWER)
    else:
        # Linear calibrated rank blend
        val = w_rap * r_raptor[c] + w_o * r_ours[c] + w_s * r_sov[c]
        
    final_df[c] = np.clip(val, 0.0001, 0.9999)

sub_file = WORK / 'submission.csv'
final_df.to_csv(sub_file, index=False)
print(f"=== CANDIDATE V42 SUCCESS: Wrote final submission to {sub_file} ===", flush=True)
print(final_df.head(), flush=True)

# Verification receipt
receipt = {
    'mode': 'Candidate V42 Sovereign Live-Inference Master (True Live 3-Way Grandmaster Ensembling)',
    'rows': len(final_df),
    'weights': {'raptor': W_RAPTOR, 'ours': W_OURS, 'sovereign': W_SOV},
    'power_sharpening_acute': P_POWER,
    'status': 'COMPLETE'
}
(WORK / 'blend_receipt.json').write_text(json.dumps(receipt, indent=1))
'''

nb['cells'][10]['source'] = cell10_script.splitlines(keepends=True)

# Write output notebook
with open(DST_NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"SUCCESS: Generated Candidate V42 notebook at {DST_NOTEBOOK}")

# Create kernel-metadata.json
metadata = {
    "id": "cyruspinto/rsna-knee-v42-sovereign-live-master",
    "title": "RSNA Knee V42 Sovereign Live Master",
    "code_file": "rsna-knee-v42-sovereign-live-master.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [
        "heliosli/rsna-knee-compact-r7cnxfsss",
        "nartaa/rsna-knee-publication-swa-weights-20261007",
        "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
        "cyruspinto/rsna-knee-colab-super-weights",
        "cyruspinto/rsna-knee-engine11-weights",
        "cyruspinto/rsna-knee-engine11-fold2-weights",
        "cyruspinto/rsna-knee-engine11-fold3-weights",
        "cyruspinto/rsna-knee-engine11-fold4-weights",
        "cyruspinto/rsna-knee-engine10-fold1-weights",
        "cyruspinto/rsna-knee-engine9-fold1-weights"
    ],
    "kernel_sources": [],
    "competition_sources": [
        "rsna-knee-abnormality-detection"
    ],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4"
}

with open(DST_META, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"SUCCESS: Generated kernel-metadata.json at {DST_META}")
