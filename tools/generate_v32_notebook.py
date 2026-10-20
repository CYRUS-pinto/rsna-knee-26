#!/usr/bin/env python3
"""
generate_v32_notebook.py
Assembles the complete Candidate V32 Tri-Engine Non-Fat-Suppressed Medical Master notebook.
Integrates:
1. Raptor 384px + 224px (Fat-Suppressed whole-knee diffuse attention)
2. ConvNeXt R7 (Fat-Suppressed dense sliding-window, Stride-2, 4 workers)
3. RadImageNet ResNet-50 E11 (Non-Fat-Suppressed anatomical contrast on 4 planes)
4. Rank-Power Ensembling (p = 1.8) with Prevalence-Shift Invariant Calibration
"""

import json
from pathlib import Path

BASE_NB_PATH = Path("submissions/v30_lightning_power_master/rsna-knee-v30-lightning-power-master.ipynb")
TARGET_NB_PATH = Path("submissions/v32_tri_engine_medical_master/rsna-knee-v32-tri-engine-medical-master.ipynb")

with open(BASE_NB_PATH, "r", encoding="utf-8") as f:
    base_nb = json.load(f)

# Cell 0: Updated Markdown Description
cell0_text = (
    "# RSNA Knee Candidate V32: Tri-Engine Non-Fat-Suppressed Medical Master\n\n"
    "**Grandmaster Tri-Engine Orthogonal Ensemble** combining:\n"
    "1. **Engine 1:** Raptor CoAtNet MultiScale (384px + 224px) on fluid-sensitive Fat-Suppressed sequences.\n"
    "2. **Engine 2:** ConvNeXt R7 Compact with Stride-2 Window Pruning (`eval_stride = 2`, 262 windows) and 4-worker DataLoader.\n"
    "3. **Engine 3:** RadImageNet ResNet-50 E11 Medical Backbone on 4 anatomical planes (3 Non-Fat-Suppressed: `SAG_NOFS`, `COR_NOFS`, `AX_NOFS` + 1 Fat-Suppressed anchor: `SAG_FS`) with 5-fold FoundationQueryHead.\n"
    "4. **Engine 4:** Percentile Rank-Power Ensembling ($p = 1.8$) with audited per-target weights and Prevalence-Shift Invariance.\n"
)
base_nb["cells"][0]["source"] = cell0_text

# Cell 8 in V30 was the blending cell. We will insert RadImageNet cells before the final blend!
# Cell 8 will become: %%writefile /tmp/rad_run.py
# Cell 9 will be: execution of /tmp/rad_run.py
# Cell 10 will be: Tri-Engine Rank-Power Blending

rad_run_code = r"""%%writefile /tmp/rad_run.py
import os, sys, glob, time, json, gc, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

print("Starting RadImageNet Non-Fat-Suppressed E11 Engine...", flush=True)
T0 = time.time()
WORK = Path("/kaggle/working")
DATA = Path("/kaggle/input/rsna-knee-abnormality-detection")
if not DATA.exists():
    for p in Path("/kaggle/input").glob("*/test.csv"):
        DATA = p.parent; break

test = pd.read_csv(DATA / "test.csv", dtype={"StudyInstanceUID": str})
test_ids = test["StudyInstanceUID"].tolist()
n_studies = len(test_ids)
print(f"Loaded {n_studies} test studies", flush=True)

TARGETS = [
    "ACL", "Baker's", "Contusion", "Effusion", "Fracture",
    "Lateral Meniscus", "Lateral OA", "MCL",
    "Medial Meniscus", "Medial OA", "PF OA", "Synovitis"
]

E11_SLOTS = [
    ("SAG_NOFS", "Sagittal", False),
    ("COR_NOFS", "Coronal", False),
    ("AX_NOFS", "Axial", False),
    ("SAG_FS", "Sagittal", True),
]
N_SLOT = len(E11_SLOTS)
CACHE_SLICES = 8
TOKEN_DIM = 2048
HEAD_DIM = 512
IMG_SIZE = 224

# Locate Checkpoints
def find_file(name):
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            return Path(root) / name
    return None

resnet_path = find_file("ResNet50.pt")
heads_path = find_file("v52_e11_heads.pt")
if resnet_path is None or heads_path is None:
    print(f"WARNING: RadImageNet checkpoints not found! (resnet: {resnet_path}, heads: {heads_path})", flush=True)
    # Output neutral fallback
    dummy_df = pd.DataFrame(0.5, index=test_ids, columns=TARGETS)
    dummy_df.index.name = "StudyInstanceUID"
    dummy_df.reset_index().to_csv(WORK / "rad_raw.csv", index=False)
    sys.exit(0)

# Build RadImageNet ResNet50
from torchvision.models import resnet50
class RadImageNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    def forward(self, x):
        return self.backbone(x).mean(dim=(2, 3))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
encoder = RadImageNetEncoder()
encoder_state = torch.load(resnet_path, map_location="cpu", weights_only=True)
encoder.load_state_dict(encoder_state, strict=True)
encoder.eval().to(device)
for p in encoder.parameters():
    p.requires_grad_(False)

class FoundationQueryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(TOKEN_DIM), nn.Linear(TOKEN_DIM, HEAD_DIM), nn.GELU())
        self.plane = nn.Parameter(torch.randn(N_SLOT, HEAD_DIM) * .01)
        self.position = nn.Parameter(torch.randn(CACHE_SLICES, HEAD_DIM) * .01)
        self.query = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * .02)
        self.attn = nn.MultiheadAttention(HEAD_DIM, 8, dropout=.10, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(HEAD_DIM * 4), nn.Linear(HEAD_DIM * 4, HEAD_DIM),
            nn.GELU(), nn.Dropout(.15),
        )
        self.weight = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * .02)
        self.bias = nn.Parameter(torch.zeros(len(TARGETS)))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), N_SLOT, CACHE_SLICES, HEAD_DIM)
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        key_padding = mask <= 0
        all_empty = key_padding.all(1)
        if all_empty.any():
            key_padding = key_padding.clone()
            key_padding[all_empty, 0] = False
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(query, token, token, key_padding_mask=key_padding, need_weights=False)[0]
        denom = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdims=True) / denom
        mean = mean.expand(-1, len(TARGETS), -1)
        fused = self.fuse(torch.cat([attended, mean, torch.abs(attended - mean), attended * mean], -1))
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias

heads_payload = torch.load(heads_path, map_location="cpu", weights_only=True)
head_models = []
for record in heads_payload["folds"]:
    m = FoundationQueryHead().to(device)
    m.load_state_dict(record["state_dict"], strict=True)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    head_models.append(m)

# Read metadata and match slots
test_series = pd.read_csv(DATA / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
desc = test_series["SeriesDescription"].fillna("").str.lower()
fs_opts = test_series["ScanOptions"].fillna("").str.upper()
is_fs = desc.str.contains("fs|fat|fatsat") | fs_opts.str.contains("FAT|SAT")
test_series["fatsat"] = is_fs

study_groups = test_series.groupby("StudyInstanceUID")
all_features = np.zeros((n_studies, N_SLOT * CACHE_SLICES, TOKEN_DIM), dtype=np.float32)
all_masks = np.zeros((n_studies, N_SLOT * CACHE_SLICES), dtype=np.float32)

def read_dicom_slices(series_dir, n_slices=8):
    files = sorted(glob.glob(f"{series_dir}/*.dcm"))
    if not files:
        return None
    n = len(files)
    lo, hi = int(0.12 * (n - 1)), int(0.88 * (n - 1))
    idx = np.unique(np.linspace(lo, hi, n_slices).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slices:
        idx = np.append(idx, idx[-1])
    slices = []
    for i in idx[:n_slices]:
        try:
            ds = pydicom.dcmread(files[int(i)], force=True)
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1) or 1)
            intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
            arr = arr * slope + intercept
            p1, p99 = np.percentile(arr, [1.0, 99.0])
            arr = np.clip((arr - p1) / max(p99 - p1, 1e-5), 0, 1)
            im = Image.fromarray((arr * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            slices.append(np.array(im, dtype=np.float32) / 127.5 - 1.0)
        except Exception:
            slices.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))
    return np.stack(slices)

print("Extracting slices and computing RadImageNet features...", flush=True)
batch_slices = []
batch_locs = []

for s_idx, uid in enumerate(test_ids):
    if uid not in study_groups.groups:
        continue
    g = study_groups.get_group(uid)
    for slot_idx, (sname, plane, req_fs) in enumerate(E11_SLOTS):
        cand = g[(g["Anatomical_Plane"] == plane) & (g["fatsat"] == req_fs)]
        if cand.empty:
            continue
        ser_uid = cand.iloc[0]["SeriesInstanceUID"]
        sdir = DATA / "test_series" / uid / ser_uid
        if not sdir.is_dir():
            continue
        sl_arr = read_dicom_slices(sdir, CACHE_SLICES)
        if sl_arr is not None:
            for sl_i in range(CACHE_SLICES):
                batch_slices.append(sl_arr[sl_i])
                batch_locs.append((s_idx, slot_idx * CACHE_SLICES + sl_i))

if batch_slices:
    batch_slices = np.stack(batch_slices)
    n_total = len(batch_slices)
    print(f"Encoding {n_total} slices via RadImageNet ResNet50...", flush=True)
    bs = 128
    with torch.inference_mode():
        for b0 in range(0, n_total, bs):
            b_x = torch.from_numpy(batch_slices[b0:b0+bs]).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
            with torch.autocast("cuda", enabled=device.type=="cuda"):
                feat = encoder(b_x).float().cpu().numpy()
            for k in range(len(feat)):
                st_i, tok_i = batch_locs[b0 + k]
                all_features[st_i, tok_i] = feat[k]
                all_masks[st_i, tok_i] = 1.0

print("Evaluating FoundationQueryHead 5-fold models...", flush=True)
all_preds = []
with torch.inference_mode():
    x_t = torch.from_numpy(all_features).to(device)
    m_t = torch.from_numpy(all_masks).to(device)
    for head in head_models:
        with torch.autocast("cuda", enabled=device.type=="cuda"):
            logits = head(x_t, m_t)
            probs = torch.sigmoid(logits).float().cpu().numpy()
            all_preds.append(probs)

rad_preds = np.mean(all_preds, axis=0)
rad_df = pd.DataFrame(rad_preds, index=test_ids, columns=TARGETS)
rad_df.index.name = "StudyInstanceUID"
rad_df.reset_index().to_csv(WORK / "rad_raw.csv", index=False)
print(f"RadImageNet E11 inference complete in {time.time() - T0:.1f}s. Saved to rad_raw.csv", flush=True)
"""

rad_exec_code = r"""# Stage 2b: execute RadImageNet Non-Fat-Suppressed Medical Backbone
import subprocess, time
T0_RAD = time.time()
print("Launching RadImageNet E11 Non-Fat-Suppressed Medical Engine...", flush=True)
ret = subprocess.run(["python", "/tmp/rad_run.py"], capture_output=True, text=True)
print(ret.stdout)
if ret.returncode != 0:
    print("WARNING: RadImageNet E11 exited with error:", ret.stderr)
RAD_SECONDS = time.time() - T0_RAD
print(f"RadImageNet stage elapsed: {RAD_SECONDS:.1f}s", flush=True)
"""

tri_engine_blend_code = r"""# Stage 3: Tri-Engine Rank-Power Ensembling (Raptor + ConvNeXt R7 + RadImageNet Non-FS)
# Audited Orthogonal Integration:
# 1. Primary Fat-Suppressed Composite: Raptor + ConvNeXt R7 (Specialist Calibrated)
# 2. Non-Fat-Suppressed Lift: RadImageNet E11 lifts Lateral Meniscus (+0.055), Synovitis (+0.039), Bone Contusion (+0.035)
# 3. Percentile Rank-Power Ensembling (p = 1.8) ensuring strict Prevalence Shift Invariance

W_RAPTOR_C = {
    'Lateral Meniscus': 0.15,
    'Medial OA': 0.20,
    'Lateral OA': 0.20,
    'PF OA': 0.60,
    'Synovitis': 0.80,
    "Baker's": 0.35,
    'ACL': 0.70,
    'MCL': 0.65,
    'Medial Meniscus': 0.50,
    'Contusion': 0.50,
    'Effusion': 0.50,
    'Fracture': 0.50
}

# Audited RadImageNet Non-FS weights from E11 Contract (ADR-033)
# Zero weight on Baker's and Fracture (fluid-sensitive fat-suppressed is already optimal)
W_RAD = {
    'Lateral Meniscus': 0.35,
    'Synovitis': 0.30,
    'Contusion': 0.30,
    'Lateral OA': 0.25,
    'MCL': 0.20,
    'ACL': 0.20,
    'Medial OA': 0.20,
    'PF OA': 0.20,
    'Medial Meniscus': 0.20,
    'Effusion': 0.15,
    "Baker's": 0.00,
    'Fracture': 0.00
}

test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
ss = DATA / 'sample_submission.csv'
ours = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})
cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else list(ours.columns)
labels = cols[1:]
raw = pd.read_csv(WORK / 'raptor_raw.csv', dtype={'StudyInstanceUID': str})
raw224 = pd.read_csv(WORK / 'raptor224_raw.csv', dtype={'StudyInstanceUID': str})
rad_raw_path = WORK / 'rad_raw.csv'
rad_exists = rad_raw_path.is_file()
rad = pd.read_csv(rad_raw_path, dtype={'StudyInstanceUID': str}) if rad_exists else pd.DataFrame(0.5, index=range(len(test_ids)), columns=['StudyInstanceUID'] + labels)

ours = ours.set_index('StudyInstanceUID').loc[test_ids, labels]
raw = raw.set_index('StudyInstanceUID').loc[test_ids, labels]
raw224 = raw224.set_index('StudyInstanceUID').loc[test_ids, labels]
rad = rad.set_index('StudyInstanceUID').loc[test_ids, labels]

n = len(test_ids)
r384_fail = (raw.values == 0.5).all(axis=1)
r224_fail = (raw224.values == 0.5).all(axis=1)
rap_fail = r384_fail & r224_fail
our_fail = (ours.values == 0.5).all(axis=1)
rad_fail = (rad.values == 0.5).all(axis=1)

def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values

def pct_on(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any():
        r[ok] = pct(np.asarray(v, dtype=np.float64)[ok])
    return r

out = np.zeros((n, len(labels)))
p_power = 1.8

for j, c in enumerate(labels):
    # Raptor MultiScale Composite
    r384, r224 = pct_on(raw[c].values, ~r384_fail), pct_on(raw224[c].values, ~r224_fail)
    rr = np.where(r384_fail, r224, np.where(r224_fail, r384, 0.65 * r384 + 0.35 * r224))
    ro = pct_on(ours[c].values, ~our_fail)
    
    # Primary Blend (Raptor + ConvNeXt)
    wr = W_RAPTOR_C.get(c, 0.50)
    rr_safe = np.clip(np.nan_to_num(rr, nan=0.5), 0.0, 1.0)
    ro_safe = np.clip(np.nan_to_num(ro, nan=0.5), 0.0, 1.0)
    primary = (wr * (rr_safe ** p_power) + (1.0 - wr) * (ro_safe ** p_power)) ** (1.0 / p_power)
    primary[rap_fail] = ro_safe[rap_fail]
    primary[our_fail] = rr_safe[our_fail]
    
    # RadImageNet Non-FS Medical Injection
    w_rad = W_RAD.get(c, 0.0)
    if w_rad > 0.0 and rad_exists and not rad_fail.all():
        r_rad = pct_on(rad[c].values, ~rad_fail)
        r_rad_safe = np.clip(np.nan_to_num(r_rad, nan=0.5), 0.0, 1.0)
        final_col = (1.0 - w_rad) * primary + w_rad * r_rad_safe
        final_col[rad_fail] = primary[rad_fail]
    else:
        final_col = primary
        
    out[:, j] = final_col

sub = pd.DataFrame(out.astype(np.float32), columns=labels)
sub.insert(0, 'StudyInstanceUID', test_ids)
sub = sub[cols]
assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()
assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv(WORK / 'submission.csv', index=False)

receipt = {
    'mode': 'Candidate V32 Tri-Engine Non-FS Medical Master (Raptor + R7 + RadImageNet E11)',
    'rows': n,
    'power_p': p_power,
    'w_rad': W_RAD,
    'w_raptor_c': W_RAPTOR_C,
    'total_seconds': round(time.time() - NB_T0, 1)
}
(WORK / 'receipt_v32.json').write_text(json.dumps(receipt, indent=2))
print("NOTEBOOK_COMPLETE", json.dumps(receipt), flush=True)
display(sub.head())
"""

# Append cells
cells = base_nb["cells"][:8] # keep 0 to 7
# Cell 8: RadImageNet code
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": rad_run_code
})
# Cell 9: RadImageNet execution
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": rad_exec_code
})
# Cell 10: Final Master Blend
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": tri_engine_blend_code
})

base_nb["cells"] = cells

TARGET_NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(TARGET_NB_PATH, "w", encoding="utf-8") as f:
    json.dump(base_nb, f, indent=1)

print(f"Successfully generated Candidate V32 notebook at {TARGET_NB_PATH} with {len(cells)} cells.")
