import json
from pathlib import Path

dst_dir = Path('scratch/e11_diagnostic_kernel')
dst_dir.mkdir(parents=True, exist_ok=True)

meta = {
  "id": "cyruspinto/rsna-knee-e11-100-study-diagnostic",
  "title": "RSNA Knee E11 100 Study Diagnostic OOF Reproduction",
  "code_file": "rsna-knee-e11-100-study-diagnostic.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "dataset_sources": [
    "antoinegg1/rsna-knee-e11-diverse-heads-v20",
    "marwanmath/resnet-50-radimagenet-marwan"
  ],
  "kernel_sources": [],
  "competition_sources": [
    "rsna-knee-abnormality-detection"
  ],
  "machine_shape": "NvidiaTeslaT4"
}
(dst_dir / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

script_code = """import os, sys, glob, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import torch
import torch.nn as nn
from scipy.stats import spearmanr

print("Starting E11 100-Study Diagnostic Kernel...", flush=True)

DATA = None
for cand in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"), Path("/kaggle/input/rsna-knee-abnormality-detection")]:
    if (cand / "train.csv").is_file():
        DATA = cand
        break
assert DATA is not None, "train.csv not found!"

# Load OOF truth
oof_path = Path("/kaggle/input/rsna-knee-e11-diverse-heads-v20/v52_e11_oof.csv")
assert oof_path.is_file(), "v52_e11_oof.csv not found!"
oof_df = pd.read_csv(oof_path, dtype={"StudyInstanceUID": str})

TARGETS = [
    "ACL", "Baker's", "Contusion", "Effusion", "Fracture",
    "Lateral Meniscus", "Lateral OA", "MCL",
    "Medial Meniscus", "Medial OA", "PF OA", "Synovitis"
]

# Sample 100 training studies with fixed seed, stratified by Axial NOFS presence
np.random.seed(42)
train_series = pd.read_csv(DATA / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
train_series["fatsat"] = pd.to_numeric(train_series["Fat_Suppression"], errors="coerce").fillna(0).astype(int) == 1

# Studies with Axial NOFS vs without
ax_nofs_studies = set(train_series[(train_series["Anatomical_Plane"] == "Axial") & (~train_series["fatsat"])]["StudyInstanceUID"].unique())
all_train_uids = oof_df["StudyInstanceUID"].values
uids_with_ax = [u for u in all_train_uids if u in ax_nofs_studies]
uids_without_ax = [u for u in all_train_uids if u not in ax_nofs_studies]

# 50 with Axial NOFS, 50 without Axial NOFS
sample_uids = np.concatenate([
    np.random.choice(uids_with_ax, size=50, replace=False),
    np.random.choice(uids_without_ax, size=50, replace=False)
])
np.random.shuffle(sample_uids)
sample_oof = oof_df.set_index("StudyInstanceUID").loc[sample_uids]
study_groups = train_series.groupby("StudyInstanceUID")

# Load model weights
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torchvision.models import resnet50
class RadImageNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
    def forward(self, x):
        return self.backbone(x).mean(dim=(2, 3))

resnet_path = Path("/kaggle/input/resnet-50-radimagenet-marwan/ResNet50.pt")
encoder = RadImageNetEncoder()
encoder.load_state_dict(torch.load(resnet_path, map_location="cpu", weights_only=True), strict=True)
encoder.eval().to(device)

heads_path = Path("/kaggle/input/rsna-knee-e11-diverse-heads-v20/v52_e11_heads.pt")
heads_payload = torch.load(heads_path, map_location="cpu", weights_only=True)

class FoundationQueryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 512), nn.GELU())
        self.plane = nn.Parameter(torch.randn(4, 512) * .01)
        self.position = nn.Parameter(torch.randn(8, 512) * .01)
        self.query = nn.Parameter(torch.randn(len(TARGETS), 512) * .02)
        self.attn = nn.MultiheadAttention(512, 8, dropout=.10, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(512 * 4), nn.Linear(512 * 4, 512),
            nn.GELU(), nn.Dropout(.15),
        )
        self.weight = nn.Parameter(torch.randn(len(TARGETS), 512) * .02)
        self.bias = nn.Parameter(torch.zeros(len(TARGETS)))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), 4, 8, 512)
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

head_models = []
for record in heads_payload["folds"]:
    m = FoundationQueryHead().to(device)
    m.load_state_dict(record["state_dict"], strict=True)
    m.eval()
    head_models.append(m)

# Standard Slot Definitions
E11_SLOTS = [
    ("SAG_NOFS", "Sagittal", False),
    ("COR_NOFS", "Coronal", False),
    ("AX_NOFS", "Axial", False),
    ("SAG_FS", "Sagittal", True),
]

# Negative Control: Swapped Routing (FS <-> NOFS)
SWAPPED_SLOTS = [
    ("SAG_FS_SWAPPED", "Sagittal", True),
    ("COR_FS_SWAPPED", "Coronal", True),
    ("AX_FS_SWAPPED", "Axial", True),
    ("SAG_NOFS_SWAPPED", "Sagittal", False),
]

def read_dicom_slices(series_dir, n_slices=8):
    files = sorted(glob.glob(f"{series_dir}/*.dcm"))
    if not files: return None
    n = len(files)
    lo, hi = int(0.12 * (n - 1)), int(0.88 * (n - 1))
    idx = np.unique(np.linspace(lo, hi, n_slices).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slices: idx = np.append(idx, idx[-1])
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
            im = Image.fromarray((arr * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
            slices.append(np.array(im, dtype=np.float32) / 127.5 - 1.0)
        except Exception:
            slices.append(np.zeros((224, 224), dtype=np.float32))
    return np.stack(slices)

def extract_study_features(uid, slots_config):
    g = study_groups.get_group(uid)
    feats = np.zeros((1, 32, 2048), dtype=np.float32)
    masks = np.zeros((1, 32), dtype=np.float32)
    for slot_idx, (sname, plane, req_fs) in enumerate(slots_config):
        cand = g[(g["Anatomical_Plane"] == plane) & (g["fatsat"] == req_fs)]
        if cand.empty: continue
        
        # Deterministic tie-breaker: if multiple series, pick the one with most slices (exact training pick_slots logic)
        if len(cand) > 1:
            slice_counts = []
            for s_uid in cand["SeriesInstanceUID"]:
                p = DATA / "train_series" / uid / s_uid
                slice_counts.append(len(os.listdir(p)) if p.is_dir() else 0)
            cand = cand.iloc[np.argmax(slice_counts):np.argmax(slice_counts)+1]

        sdir = DATA / "train_series" / uid / cand.iloc[0]["SeriesInstanceUID"]
        if not sdir.is_dir(): continue
        sl_arr = read_dicom_slices(sdir, 8)
        if sl_arr is not None:
            with torch.inference_mode():
                b_x = torch.from_numpy(sl_arr).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
                with torch.autocast("cuda", enabled=device.type=="cuda"):
                    f = encoder(b_x).float().cpu().numpy()
            feats[0, slot_idx*8:(slot_idx+1)*8] = f
            masks[0, slot_idx*8:(slot_idx+1)*8] = 1.0
    return feats, masks

print("Running 100-study diagnostic (Held-out fold heads)...", flush=True)
t_start = time.time()
preds_heldout = []
preds_swapped = []

for uid in sample_uids:
    fold_idx = int(sample_oof.loc[uid, "fold"])
    
    # 1. Standard E11 Routing
    feats_std, masks_std = extract_study_features(uid, E11_SLOTS)
    with torch.inference_mode():
        x_t = torch.from_numpy(feats_std).to(device)
        m_t = torch.from_numpy(masks_std).to(device)
        p_std = torch.sigmoid(head_models[fold_idx](x_t, m_t)).cpu().numpy()[0]
        preds_heldout.append(p_std)

    # 2. Negative Control (Swapped Routing)
    feats_swap, masks_swap = extract_study_features(uid, SWAPPED_SLOTS)
    with torch.inference_mode():
        x_sw = torch.from_numpy(feats_swap).to(device)
        m_sw = torch.from_numpy(masks_swap).to(device)
        p_swap = torch.sigmoid(head_models[fold_idx](x_sw, m_sw)).cpu().numpy()[0]
        preds_swapped.append(p_swap)

total_time = time.time() - t_start
ms_per_study = (total_time / 100.0) * 1000.0
print(f"100 studies processed in {total_time:.1f}s ({ms_per_study:.1f} ms/study)", flush=True)

preds_df = pd.DataFrame(preds_heldout, index=sample_uids, columns=TARGETS)
swapped_df = pd.DataFrame(preds_swapped, index=sample_uids, columns=TARGETS)

corrs_std = {}
corrs_swap = {}
for c in TARGETS:
    r_std, _ = spearmanr(preds_df[c].values, sample_oof[c].values)
    r_sw, _ = spearmanr(swapped_df[c].values, sample_oof[c].values)
    corrs_std[c] = float(r_std)
    corrs_swap[c] = float(r_sw)

print("\\n--- SPEARMAN CORRELATION VS OOF TRUTH ---")
print(f"{'Finding':20s} {'Standard E11':>15s} {'Negative Control (Swapped)':>28s}")
for c in TARGETS:
    print(f"  {c:18s}: {corrs_std[c]:14.4f} {corrs_swap[c]:27.4f}")

mean_std = np.mean(list(corrs_std.values()))
mean_swap = np.mean(list(corrs_swap.values()))
min_std = np.min(list(corrs_std.values()))
n_failing = sum(1 for v in corrs_std.values() if v < 0.70)

print(f"\\nSTANDARD E11 MEAN SPEARMAN: {mean_std:.4f} (Min: {min_std:.4f}, Labels < 0.70: {n_failing})")
print(f"NEGATIVE CONTROL MEAN SPEARMAN: {mean_swap:.4f}")

receipt = {
    "total_seconds": round(total_time, 2),
    "ms_per_study": round(ms_per_study, 1),
    "mean_spearman_standard": round(mean_std, 4),
    "mean_spearman_negative_control": round(mean_swap, 4),
    "min_spearman_standard": round(min_std, 4),
    "n_labels_failing_bar": n_failing,
    "per_label_standard": corrs_std,
    "per_label_negative_control": corrs_swap
}
Path("/kaggle/working/diagnostic_receipt.json").write_text(json.dumps(receipt, indent=2))
print("DIAGNOSTIC_COMPLETE", flush=True)
"""

nb = {
    "cells": [
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [script_code]}
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 4
}
(dst_dir / 'rsna-knee-e11-100-study-diagnostic.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
print("Successfully created diagnostic kernel at:", dst_dir / 'rsna-knee-e11-100-study-diagnostic.ipynb')
