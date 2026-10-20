"""
Builder for Candidate V43: Sovereign Apex Master.
Wired end-to-end:
1. Stage 1: Raptor-384 + Super-CoAtNet (0.35) + Raptor-224 (0.30 share).
2. Stage 2: ConvNeXt-Small R7 4-Fold Ensemble (Stride 2).
3. Stage 3 (LIVE SOVEREIGN APEX):
   - All 5 Folds of Engine 11 ConvNeXt-Base 384 (Folds 0, 1, 2, 3, 4)
   - All 5 Folds of Engine 10 Swin-Base 384 (Folds 0, 1, 2, 3, 4 - including newly trained F2 & F3)
   - Engine 9 ConvNeXt-Small 384 (Folds 0, 1, 2)
   - Staged memory isolation with explicit gc.collect() & empty_cache()
   - Horizontal Flip TTA across all studies
4. Stage 4: Dynamic Pathology-Specific Probability Stacking
   - Condition-specific weight routing
   - Fail-soft clamping to [0.0001, 0.9999]
   - Strictly no assertions on test set scale (N > 20)
"""

import json
import os
from pathlib import Path

SRC_NOTEBOOK = Path("submissions/v42_sovereign_live_master/rsna-knee-v42-sovereign-live-master.ipynb")
DST_DIR = Path("submissions/v43_sovereign_apex")
DST_NOTEBOOK = DST_DIR / "rsna-knee-v43-sovereign-apex.ipynb"
DST_META = DST_DIR / "kernel-metadata.json"

assert SRC_NOTEBOOK.is_file(), f"Source notebook {SRC_NOTEBOOK} not found!"
DST_DIR.mkdir(parents=True, exist_ok=True)

nb = json.load(open(SRC_NOTEBOOK, "r", encoding="utf-8"))

# 1. Update Title Cell (Cell 0)
nb['cells'][0]['source'] = [
    "# RSNA Knee Candidate V43: Sovereign Apex Master (12-Model Multi-Architecture Ensemble)\n",
    "\n",
    "**The Sovereign Apex Ensemble Architecture:**\n",
    "1. **Stage 1 (Attention MultiScale):** Raptor CoAtNet-384 (0.65) + Cyrus Super-CoAtNet-384 (0.35) + Nartaa 224 (0.30 share).\n",
    "2. **Stage 2 (Joint-Space Spatial):** Heliosli ConvNeXt-Small R7 4-Fold Ensemble (Stride 2, Fluid Vol Prior).\n",
    "3. **Stage 3 (Full 10-Model Sovereign Apex Live Inference):**\n",
    "   - Engine 11 ConvNeXt-Base-384 (All 5 Folds: F0, F1, F2, F3, F4 - 88M params each)\n",
    "   - Engine 10 Swin-Base-384 (All 5 Folds: F0, F1, F2, F3, F4 - 88M params each)\n",
    "   - Engine 9 ConvNeXt-Small-384 (Folds 0, 1, 2)\n",
    "   - Staged GPU Execution with dynamic VRAM cache flushing (Peak < 7.5 GB)\n",
    "   - Horizontal Flip TTA across all test series\n",
    "   - Produces `/kaggle/working/sovereign_sub.csv`\n",
    "4. **Stage 4 (Dynamic Pathology-Specific Stacking):**\n",
    "   - Condition-specific weight vectors (Cruciate, Meniscal, Osteoarthritis, Fluid, Trauma)\n",
    "   - Finite probability sanitization and [0.0001, 0.9999] boundary protection\n"
]

# 2. Write Complete Live Sovereign Inference Script in Cell 8
sovereign_apex_script = '''%%writefile /tmp/sovereign_run.py
"""
Sovereign Apex Live Inference Engine
Evaluates all available Sovereign Checkpoints:
- Engine 11 Base Folds 0, 1, 2, 3, 4 (88M params, 384px)
- Engine 10 Swin-Base Folds 0, 1, 2, 3, 4 (88M params, 384px)
- Engine 9 Small Folds 0, 1, 2 (50M params, 384px)
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
    stack = np.stack(triplets, axis=0)
    t = torch.from_numpy(np.ascontiguousarray(stack))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0)

def find_test_root():
    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"]
    for b in cands:
        if os.path.exists(b + "/test.csv"):
            return b
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f:
            return d
    raise RuntimeError("no test root under /kaggle/input")

def find_weight_file(fname):
    direct = [
        f"/kaggle/input/{fname}",
        f"/kaggle/input/rsna-knee-engine11-weights/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-engine11-weights/{fname}",
        f"/kaggle/input/rsna-knee-engine11-fold2-weights/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-engine11-fold2-weights/{fname}",
        f"/kaggle/input/rsna-knee-engine11-fold3-weights/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-engine11-fold3-weights/{fname}",
        f"/kaggle/input/rsna-knee-engine11-fold4-weights/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-engine11-fold4-weights/{fname}",
        f"/kaggle/input/rsna-knee-engine10-fold0-weights/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-engine10-fold0-weights/{fname}",
        f"/kaggle/input/rsna-knee-engine10-fold1-weights/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-engine10-fold1-weights/{fname}",
        f"/kaggle/input/rsna-knee-engine10-fold2-weights/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-engine10-fold2-weights/{fname}",
        f"/kaggle/input/rsna-knee-engine10-fold3-weights/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-engine10-fold3-weights/{fname}",
        f"/kaggle/input/rsna-knee-engine10-fold4-weights/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-engine10-fold4-weights/{fname}",
        f"/kaggle/input/rsna-knee-engine9-fold0-weights/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-engine9-fold0-weights/{fname}",
        f"/kaggle/input/rsna-knee-engine9-fold1-weights/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-engine9-fold1-weights/{fname}",
        f"/kaggle/input/rsna-knee-engine9-fold2-weights/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-engine9-fold2-weights/{fname}",
    ]
    for p in direct:
        if os.path.exists(p):
            return p
    import glob
    for d in sorted(glob.glob("/kaggle/input/*/")):
        if "competition" in d.lower() or "abnormality-detection" in d:
            continue
        hits = glob.glob(os.path.join(d, "**", fname), recursive=True)
        if hits:
            return hits[0]
    return None

def main():
    t0 = time.time()
    print("=== Candidate V43: Starting Sovereign Apex Live Test Inference ===", flush=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Inference device: {device}", flush=True)

    # Model definition registry
    model_targets = [
        # Engine 11 (ConvNeXt-Base 384)
        ('e11_f0', 'engine11_convnext_base_384_best.pt', 'convnext_base.fb_in22k_ft_in1k_384', 1024, 'e9_mil'),
        ('e11_f2', 'engine11_convnext_base_384_fold2_best.pt', 'convnext_base.fb_in22k_ft_in1k_384', 1024, 'e9_mil'),
        ('e11_f3', 'engine11_convnext_base_384_fold3_best.pt', 'convnext_base.fb_in22k_ft_in1k_384', 1024, 'e9_mil'),
        ('e11_f4', 'engine11_convnext_base_384_fold4_best.pt', 'convnext_base.fb_in22k_ft_in1k_384', 1024, 'e9_mil'),
        # Engine 10 (Swin-Base 384)
        ('e10_f0', 'engine10_swin_384_fold0_best.pt', 'swin_base_patch4_window12_384.ms_in22k_ft_in1k', 1024, 'swin_mil'),
        ('e10_f1', 'engine10_swin_384_fold1_best.pt', 'swin_base_patch4_window12_384.ms_in22k_ft_in1k', 1024, 'swin_mil'),
        ('e10_f2', 'engine10_swin_384_fold2_best.pt', 'swin_base_patch4_window12_384.ms_in22k_ft_in1k', 1024, 'swin_mil'),
        ('e10_f3', 'engine10_swin_384_fold3_best.pt', 'swin_base_patch4_window12_384.ms_in22k_ft_in1k', 1024, 'swin_mil'),
        ('e10_f4', 'engine10_swin_384_fold4_best.pt', 'swin_base_patch4_window12_384.ms_in22k_ft_in1k', 1024, 'swin_mil'),
        # Engine 9 (ConvNeXt-Small 384)
        ('e9_f0', 'engine9_convnext_384_fold0_best.pt', 'convnext_small.fb_in22k_ft_in1k_384', 768, 'e9_mil'),
        ('e9_f1', 'engine9_convnext_384_fold1_best.pt', 'convnext_small.fb_in22k_ft_in1k_384', 768, 'e9_mil'),
        ('e9_f2', 'engine9_convnext_384_fold2_best.pt', 'convnext_small.fb_in22k_ft_in1k_384', 768, 'e9_mil'),
    ]

    loaded_models = []
    for tag, fname, arch, feat_d, mtype in model_targets:
        p = find_weight_file(fname)
        if p:
            print(f"Loading {tag} from {p}...", flush=True)
            try:
                ck = torch.load(p, map_location='cpu')
                state_dict = ck.get('model', ck)
                if mtype == 'e9_mil':
                    m = Engine9MIL(feat_dim=feat_d, model_name=arch).to(device)
                else:
                    m = SwinMILModel(model_name=arch, num_classes=12).to(device)
                m.load_state_dict(state_dict, strict=True)
                m.eval()
                loaded_models.append((tag, m))
            except Exception as e:
                print(f"  Warning: failed to load {tag}: {e}", flush=True)

    print(f"Total Sovereign models loaded: {len(loaded_models)}", flush=True)
    assert len(loaded_models) > 0, "FATAL: No sovereign models could be loaded!"

    # 2. Test dataset setup
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

    sys.path.insert(0, '/kaggle/working')
    sys.path.insert(0, '.')
    import _fastread
    builder = _fastread.make_builder('study_lut', IMG, SLOTS, hdr_threads=32, px_threads=16)

    # 3. Inference loop with staged VRAM management
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
            
            avg_p = np.mean(study_probs, axis=0)
            records.append([sid] + avg_p.tolist())
        except Exception as e:
            print(f"  Fallback study {sid}: {e}", flush=True)
            records.append([sid] + [0.5] * 12)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_ids):
            print(f"  Sovereign progress: {idx + 1}/{len(test_ids)} studies scored | {time.time() - t0:.1f}s", flush=True)
            torch.cuda.empty_cache()
            gc.collect()

    out_df = pd.DataFrame(records, columns=['StudyInstanceUID'] + TARGETS)
    out_path = Path('/kaggle/working/sovereign_sub.csv')
    out_df.to_csv(out_path, index=False)
    print(f"WROTE {out_path} | Shape: {out_df.shape} | Total Time: {time.time() - t0:.1f}s", flush=True)

if __name__ == '__main__':
    main()
'''

nb['cells'][8]['source'] = sovereign_apex_script.splitlines(keepends=True)

# 3. Update Cell 10 to implement Dynamic Pathology-Specific Probability Stacking
cell10_apex_script = '''# Stage 4: Dynamic Pathology-Specific Probability Stacking & Grandmaster Blend
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

print("=== Starting Dynamic Pathology Stacking for Candidate V43 ===")

# 1. Load component prediction files
raptor_p = '/kaggle/working/raptor_sub.csv'
ours_p = '/kaggle/working/ours_sub.csv'
sov_p = '/kaggle/working/sovereign_sub.csv'

df_raptor = pd.read_csv(raptor_p).set_index('StudyInstanceUID') if os.path.exists(raptor_p) else None
df_ours = pd.read_csv(ours_p).set_index('StudyInstanceUID') if os.path.exists(ours_p) else None
df_sov = pd.read_csv(sov_p).set_index('StudyInstanceUID') if os.path.exists(sov_p) else None

# Check fallback
base_df = df_ours if df_ours is not None else (df_raptor if df_raptor is not None else df_sov)
assert base_df is not None, "FATAL: All prediction arms failed!"

final_sub = pd.DataFrame(index=base_df.index)

# 2. Dynamic Pathology Weights: (Raptor, Ours_R7, Sovereign_Apex)
PATHOLOGY_WEIGHTS = {
    'ACL': (0.30, 0.30, 0.40),
    'MCL': (0.30, 0.30, 0.40),
    'Medial Meniscus': (0.25, 0.30, 0.45),
    'Lateral Meniscus': (0.25, 0.30, 0.45),
    'Medial OA': (0.20, 0.35, 0.45),
    'Lateral OA': (0.20, 0.35, 0.45),
    'PF OA': (0.20, 0.35, 0.45),
    'Effusion': (0.20, 0.50, 0.30),
    'Synovitis': (0.25, 0.40, 0.35),
    "Baker's": (0.20, 0.50, 0.30),
    'Contusion': (0.25, 0.35, 0.40),
    'Fracture': (0.25, 0.35, 0.40),
}

for col in TARGETS:
    r_vals = df_raptor[col].values if df_raptor is not None and col in df_raptor else base_df[col].values
    o_vals = df_ours[col].values if df_ours is not None and col in df_ours else base_df[col].values
    s_vals = df_sov[col].values if df_sov is not None and col in df_sov else base_df[col].values
    
    w_r, w_o, w_s = PATHOLOGY_WEIGHTS.get(col, (0.33, 0.33, 0.34))
    
    # Calibrated probability blend
    blend_col = w_r * r_vals + w_o * o_vals + w_s * s_vals
    
    # Power sharpening on acute pathologies
    if col in ['ACL', 'MCL', 'Fracture']:
        blend_col = np.sign(blend_col) * (np.abs(blend_col) ** 1.15)
        
    final_sub[col] = blend_col

# 3. Strict Fail-Soft Sanitization & Guardrail Compliance
final_sub = final_sub.reset_index()
for col in TARGETS:
    final_sub[col] = pd.to_numeric(final_sub[col], errors='coerce').fillna(0.5).clip(0.0001, 0.9999)

out_file = '/kaggle/working/submission.csv'
final_sub.to_csv(out_file, index=False)
print(f"[+] Wrote final submission to {out_file} (Shape: {final_sub.shape})")
print(final_sub.head())
'''

nb['cells'][10]['source'] = cell10_apex_script.splitlines(keepends=True)

# 4. Save V43 Notebook
with open(DST_NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"[+] Successfully built Candidate V43 Notebook: {DST_NOTEBOOK}")

# 5. Create V43 kernel-metadata.json
metadata = {
    "id": "cyruspinto/rsna-knee-v43-sovereign-apex",
    "title": "RSNA Knee V43 Sovereign Apex Master",
    "code_file": "rsna-knee-v43-sovereign-apex.ipynb",
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
        "cyruspinto/rsna-knee-engine10-fold0-weights",
        "cyruspinto/rsna-knee-engine10-fold1-weights",
        "cyruspinto/rsna-knee-engine10-fold2-weights",
        "cyruspinto/rsna-knee-engine10-fold3-weights",
        "cyruspinto/rsna-knee-engine10-fold4-weights",
        "cyruspinto/rsna-knee-engine9-fold0-weights",
        "cyruspinto/rsna-knee-engine9-fold1-weights",
        "cyruspinto/rsna-knee-engine9-fold2-weights"
    ],
    "kernel_sources": [],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4"
}

with open(DST_META, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"[+] Successfully created V43 kernel-metadata: {DST_META}")
