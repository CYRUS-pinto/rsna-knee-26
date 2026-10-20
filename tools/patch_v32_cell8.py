import json

nb_path = 'submissions/v32_tri_engine_medical_master/rsna-knee-v32-tri-engine-medical-master.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

cell8_code = """%%writefile /tmp/rad_run.py
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
DATA = None
for cand in [
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
]:
    if (cand / "test_series.csv").is_file():
        DATA = cand
        break
if DATA is None:
    for p in Path("/kaggle/input").rglob("test_series.csv"):
        DATA = p.parent
        break
assert DATA is not None and (DATA / "test_series.csv").is_file(), f"Competition test data not found! Search checked: {list(Path('/kaggle/input').glob('*'))}"

test = pd.read_csv(DATA / "test.csv", dtype={"StudyInstanceUID": str})
test_ids = list(test["StudyInstanceUID"].values)
n_studies = len(test_ids)
print(f"Loaded {n_studies} test studies from {DATA / 'test.csv'}", flush=True)

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
    dummy_df = pd.DataFrame(0.5, index=test_ids, columns=TARGETS)
    dummy_df.index.name = "StudyInstanceUID"
    dummy_df.to_csv(WORK / "rad_raw.csv")
    sys.exit(0)

print(f"Found RadImageNet weights: {resnet_path}", flush=True)
print(f"Found E11 Query Heads: {heads_path}", flush=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Inference device: {device}", flush=True)

# Build ResNet50 Feature Extractor
from torchvision.models import resnet50
class RadImageNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    def forward(self, x):
        return self.backbone(x).mean(dim=(2, 3))

encoder = RadImageNetEncoder()
encoder.load_state_dict(torch.load(resnet_path, map_location="cpu", weights_only=True), strict=True)
encoder.eval().to(device)
for p in encoder.parameters(): p.requires_grad_(False)

# Build FoundationQueryHead Architecture
class FoundationQueryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(TOKEN_DIM),
            nn.Linear(TOKEN_DIM, HEAD_DIM),
            nn.GELU(),
        )
        self.plane = nn.Parameter(torch.randn(N_SLOT, HEAD_DIM) * .01)
        self.position = nn.Parameter(torch.randn(CACHE_SLICES, HEAD_DIM) * .01)
        self.query = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * .02)
        self.attn = nn.MultiheadAttention(HEAD_DIM, 8, dropout=.10, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(HEAD_DIM * 4),
            nn.Linear(HEAD_DIM * 4, HEAD_DIM),
            nn.GELU(),
            nn.Dropout(.15),
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
if "Fat_Suppression" in test_series.columns:
    test_series["fatsat"] = pd.to_numeric(test_series["Fat_Suppression"], errors="coerce").fillna(0).astype(int) == 1
else:
    desc = test_series.get("SeriesDescription", pd.Series("", index=test_series.index)).fillna("").str.lower()
    fs_opts = test_series.get("ScanOptions", pd.Series("", index=test_series.index)).fillna("").str.upper()
    test_series["fatsat"] = desc.str.contains("fs|fat|fatsat") | fs_opts.str.contains("FAT|SAT")

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
collisions_seen = 0
ties_resolved_nonzero = 0
corrupt_studies = 0

for s_idx, uid in enumerate(test_ids):
    if uid not in study_groups.groups:
        continue
    g = study_groups.get_group(uid)
    for slot_idx, (sname, plane, req_fs) in enumerate(E11_SLOTS):
        cand = g[(g["Anatomical_Plane"] == plane) & (g["fatsat"] == req_fs)]
        if cand.empty:
            continue
        if len(cand) > 1:
            collisions_seen += 1
            slice_counts = []
            for s_uid in cand["SeriesInstanceUID"]:
                p = DATA / "test_series" / uid / s_uid
                slice_counts.append(len(os.listdir(p)) if p.is_dir() else 0)
            cand = cand.copy()
            cand["_n_slices"] = slice_counts
            if max(slice_counts) > 0:
                ties_resolved_nonzero += 1
            # Exact training tie-break: sort by slice count descending; stable tie-break by SeriesInstanceUID ascending
            cand = cand.sort_values(by=["_n_slices", "SeriesInstanceUID"], ascending=[False, True])

        ser_uid = cand.iloc[0]["SeriesInstanceUID"]
        sdir = DATA / "test_series" / uid / ser_uid
        if not sdir.is_dir():
            continue
        try:
            sl_arr = read_dicom_slices(sdir, CACHE_SLICES)
        except Exception as e:
            corrupt_studies += 1
            sl_arr = None

        if sl_arr is not None:
            for sl_i in range(CACHE_SLICES):
                batch_slices.append(sl_arr[sl_i])
                batch_locs.append((s_idx, slot_idx * CACHE_SLICES + sl_i))

print(f"Tie-breaking audit: {collisions_seen} collisions checked, {ties_resolved_nonzero} resolved with verified non-zero slice counts.", flush=True)
if collisions_seen > 0 and ties_resolved_nonzero == 0:
    raise RuntimeError("CRITICAL: Directory structure mismatch! All collision candidate folders showed 0 slices.")

if corrupt_studies > max(2, int(0.05 * n_studies)):
    raise RuntimeError(f"Excessive DICOM corruption: {corrupt_studies}/{n_studies} studies failed slice decoding.")

if batch_slices:
    batch_slices = np.stack(batch_slices)
    n_total = len(batch_slices)
    print(f"Encoding {n_total} slices via RadImageNet ResNet50...", flush=True)
    bs = 64
    for b0 in range(0, n_total, bs):
        with torch.inference_mode():
            b_x = torch.from_numpy(batch_slices[b0:b0+bs]).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                f = encoder(b_x).float().cpu().numpy()
        for i, (s_idx, tok_idx) in enumerate(batch_locs[b0:b0+bs]):
            all_features[s_idx, tok_idx] = f[i]
            all_masks[s_idx, tok_idx] = 1.0

print(f"Features computed. Slot fill rate: {(all_masks.sum() / (n_studies * N_SLOT * CACHE_SLICES)):.1%}", flush=True)

# Run 5-fold FoundationQueryHeads
rad_preds = np.zeros((n_studies, len(TARGETS)), dtype=np.float32)
with torch.inference_mode():
    x_t = torch.from_numpy(all_features).to(device)
    m_t = torch.from_numpy(all_masks).to(device)
    for fold_i, model in enumerate(head_models):
        logits = model(x_t, m_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        rad_preds += probs / len(head_models)

rad_df = pd.DataFrame(rad_preds, index=test_ids, columns=TARGETS)
rad_df.index.name = "StudyInstanceUID"
rad_df.to_csv(WORK / "rad_raw.csv")
print(f"RadImageNet Non-FS E11 complete in {time.time() - T0:.1f}s. Saved {WORK / 'rad_raw.csv'}", flush=True)
"""

nb['cells'][8]['source'] = [cell8_code]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully wrote updated Cell 8 to V32!")
