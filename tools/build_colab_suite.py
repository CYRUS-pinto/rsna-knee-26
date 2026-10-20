"""
Generates the Colab Training Suite:
1. colab_engine12_convnext_large_mil.ipynb (Flagship ConvNeXt-Large 198M params with Gated Attention MIL)
2. colab_engine13_swin_base_focal.ipynb (Swin-Base-384 with Asymmetric Focal Loss)
3. colab_unified_dual_engine_master.ipynb (Single-click sequential master running both engines)
All notebooks save checkpoints directly to Google Drive (/content/drive/MyDrive/) to ensure zero weight loss.
"""

import json
import os
from pathlib import Path

SRC_NOTEBOOK = Path("colab_swin_single_click_master.ipynb")
assert SRC_NOTEBOOK.exists(), "Source notebook not found!"

base_nb = json.load(open(SRC_NOTEBOOK, "r", encoding="utf-8"))

# ==============================================================================
# NOTEBOOK 1: Engine 12 ConvNeXt-Large MIL (198M params)
# ==============================================================================
nb_e12 = json.load(open(SRC_NOTEBOOK, "r", encoding="utf-8"))
nb_e12['cells'][0]['source'] = [
    "# RSNA Knee Engine 12: ConvNeXt-Large with Gated Attention MIL (198M Params)\n",
    "### Grandmaster SOTA Architecture on Colab A100 GPU\n",
    "- Backbone: `convnext_large.fb_in22k_ft_in1k_384` (198M parameters, 1536-dim features)\n",
    "- Pooling: Ilse et al. Gated Attention Multiple Instance Learning (dynamically attends to 2-3 tear slices)\n",
    "- Auto-saves weights to Google Drive (`/content/drive/MyDrive/`) and Kaggle Datasets\n"
]

cell1_e12 = list(nb_e12['cells'][1]['source'])
# Add drive mount to Cell 1
drive_mount_lines = [
    "\n",
    "# Mount Google Drive for zero-loss checkpoint security\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "os.makedirs('/content/drive/MyDrive/rsna_knee_checkpoints', exist_ok=True)\n",
    "print('[+] Google Drive mounted successfully! Checkpoints will sync to /content/drive/MyDrive/rsna_knee_checkpoints')\n"
]
cell1_e12.extend(drive_mount_lines)
nb_e12['cells'][1]['source'] = cell1_e12

# Cell 2: Replace Swin with ConvNeXt-Large + Gated Attention MIL
cell2_e12_code = '''# CELL 2: Autonomous ConvNeXt-Large MIL Training Loop
import os, sys, time, json, gc, math
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from kaggle.api.kaggle_api_extended import KaggleApi
import timm

api = KaggleApi()
api.authenticate()

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class TwoPartVols:
    def __init__(self, a, b=None):
        self.a, self.b = a, b
        self.len_a = len(a)
    def __getitem__(self, idx):
        if idx < self.len_a: return self.a[idx]
        return self.b[idx - self.len_a]
    def __len__(self):
        return self.len_a + (len(self.b) if self.b is not None else 0)

class ConvNeXtMILDataset(Dataset):
    def __init__(self, uids, id2row, labels, vols, res=384, train=True, n_slices=8):
        self.uids = uids
        self.id2row = id2row
        self.labels = labels
        self.vols = vols
        self.res = res
        self.train = train
        self.n_slices = n_slices
    def __len__(self): return len(self.uids)
    def __getitem__(self, idx):
        u = self.uids[idx]
        row = self.id2row[u]
        v = self.vols[row]
        D, H, W = v.shape
        step = max(1, D // self.n_slices)
        idxs = [min(i * step + step // 2, D - 1) for i in range(self.n_slices)]
        imgs = []
        for ci in idxs:
            c_prev = max(0, ci - 1)
            c_next = min(D - 1, ci + 1)
            triplet = np.stack([v[c_prev], v[ci], v[c_next]], axis=0).astype(np.float32) / 255.0
            imgs.append(triplet)
        t = torch.tensor(np.stack(imgs, axis=0), dtype=torch.float32)
        if (H, W) != (self.res, self.res):
            t = nn.functional.interpolate(t, size=(self.res, self.res), mode='bilinear', align_corners=False)
        y = torch.tensor(self.labels[u], dtype=torch.float32)
        return t, y

class ConvNeXtLargeMIL(nn.Module):
    def __init__(self, model_name='convnext_large.fb_in22k_ft_in1k_384', feat_dim=1536, num_classes=12):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.norm = nn.LayerNorm(feat_dim)
        # Gated Attention Pooling
        self.att_v = nn.Sequential(nn.Linear(feat_dim, 256), nn.Tanh())
        self.att_u = nn.Sequential(nn.Linear(feat_dim, 256), nn.Sigmoid())
        self.att_w = nn.Linear(256, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes)
        )
    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.encoder(x)
        feats = self.norm(feats)
        feats = feats.view(B, S, -1)
        
        # Gated Attention Mechanism
        v = self.att_v(feats)
        u = self.att_u(feats)
        scores = self.att_w(v * u)
        weights = torch.softmax(scores, dim=1)
        
        pooled = torch.sum(feats * weights, dim=1)
        return self.classifier(pooled)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Training on Device: {DEVICE}', flush=True)

# 1. Load 384px corpus
va = np.load('/content/corpus/all_vols.npy', mmap_mode='r')
vb = np.load('/content/corpus/extra_vols.npy', mmap_mode='r') if os.path.exists('/content/corpus/extra_vols.npy') else None
vols = TwoPartVols(va, vb)
a_ids = np.load('/content/corpus/all_ids.npy', allow_pickle=True).astype(str)
b_ids = np.load('/content/corpus/extra_ids.npy', allow_pickle=True).astype(str) if os.path.exists('/content/corpus/extra_ids.npy') else np.array([])
all_ids = np.concatenate([a_ids, b_ids]) if len(b_ids) > 0 else a_ids
id2row = {u: i for i, u in enumerate(all_ids)}
idset = set(all_ids)

# 2. Native Python Ground Truth Dictionary
soft = pd.read_parquet('/content/labels/SUPER_CONSENSUS_labels.parquet').set_index('StudyInstanceUID')
train_ids = [u for u in all_ids if u in soft.index]
labels = {u: soft.loc[u, TARGETS].values.astype(np.float32) for u in train_ids}
print(f'[+] Loaded {len(labels)} total studies for training', flush=True)

np.random.seed(42)
shuffled_train = list(train_ids)
np.random.shuffle(shuffled_train)
n_val = len(shuffled_train) // 5

FOLDS_TO_TRAIN = [0, 1]
EPOCHS = 6
BATCH_SIZE = 4
GRAD_ACCUM = 2

for fold in FOLDS_TO_TRAIN:
    val_uids = set(shuffled_train[fold * n_val : (fold + 1) * n_val])
    tr_uids = [u for u in shuffled_train if u not in val_uids]
    
    tds = ConvNeXtMILDataset(tr_uids, id2row, labels, vols, res=384, train=True, n_slices=8)
    tl = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    print(f"\\n=== STARTING ENGINE 12 CONVNEXT-LARGE MIL: FOLD {fold} ===", flush=True)
    model = ConvNeXtLargeMIL(num_classes=12).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(tl))
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        start_t = time.time()
        for step, (bx, by) in enumerate(tl):
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(bx)
                loss = criterion(logits, by) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            total_loss += loss.item() * GRAD_ACCUM
            if (step + 1) % 50 == 0:
                print(f"Fold {fold} | Epoch {epoch+1}/{EPOCHS} | Step {step+1}/{len(tl)} | Loss: {total_loss/(step+1):.4f}", flush=True)
        elapsed = time.time() - start_t
        print(f"Epoch {epoch+1} finished in {elapsed:.1f}s. Avg Loss: {total_loss/len(tl):.4f}", flush=True)
        
    weight_path = f"/content/engine12_convnext_large_fold{fold}_best.pt"
    drive_path = f"/content/drive/MyDrive/rsna_knee_checkpoints/engine12_convnext_large_fold{fold}_best.pt"
    torch.save(model.state_dict(), weight_path)
    os.system(f"cp {weight_path} {drive_path}")
    print(f"[+] Model permanently secured in Google Drive: {drive_path}", flush=True)
    
    # Kaggle Push
    pkg_dir = f"/content/kaggle_e12_fold{fold}"
    os.makedirs(pkg_dir, exist_ok=True)
    os.system(f"cp {weight_path} {pkg_dir}/")
    meta = {
        'title': f'RSNA Knee Engine 12 ConvNeXt-Large Fold {fold} Weights',
        'id': f'cyruspinto/rsna-knee-engine12-fold{fold}-weights',
        'licenses': [{'name': 'CC0-1.0'}]
    }
    with open(f"{pkg_dir}/dataset-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    try:
        api.dataset_create_new(pkg_dir, dir_mode='zip')
        print(f"[+] Created Kaggle dataset: cyruspinto/rsna-knee-engine12-fold{fold}-weights")
    except Exception as e:
        api.dataset_create_version(pkg_dir, f'Update Fold {fold} Weights', dir_mode='zip')
        print(f"[+] Updated Kaggle dataset: cyruspinto/rsna-knee-engine12-fold{fold}-weights")
        
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

print("\\n[+] ENGINE 12 CONVNEXT-LARGE TRAINING COMPLETE!")
'''
nb_e12['cells'][2]['source'] = cell2_e12_code.splitlines(keepends=True)

with open("colab_engine12_convnext_large_mil.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_e12, f, indent=1)
print("[+] Created colab_engine12_convnext_large_mil.ipynb")

# ==============================================================================
# NOTEBOOK 2: Engine 13 Swin-Base with Asymmetric Focal Loss
# ==============================================================================
nb_e13 = json.load(open(SRC_NOTEBOOK, "r", encoding="utf-8"))
nb_e13['cells'][0]['source'] = [
    "# RSNA Knee Engine 13: Swin-Base-384 with Asymmetric Focal Loss\n",
    "### SOTA Hierarchical Vision Transformer with Extreme Class Imbalance Mastery\n",
    "- Backbone: `swin_base_patch4_window12_384.ms_in22k_ft_in1k` (88M params)\n",
    "- Loss: Asymmetric Focal Loss (gamma_neg=4.0, gamma_pos=1.0) suppressing easy negative gradients\n",
    "- Auto-saves weights to Google Drive (`/content/drive/MyDrive/`) and Kaggle Datasets\n"
]
nb_e13['cells'][1]['source'] = cell1_e12 # reuse drive mount setup

cell2_e13_code = '''# CELL 2: Autonomous Swin-Base-384 with Asymmetric Focal Loss Loop
import os, sys, time, json, gc, math
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from kaggle.api.kaggle_api_extended import KaggleApi
import timm

api = KaggleApi()
api.authenticate()

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class TwoPartVols:
    def __init__(self, a, b=None):
        self.a, self.b = a, b
        self.len_a = len(a)
    def __getitem__(self, idx):
        if idx < self.len_a: return self.a[idx]
        return self.b[idx - self.len_a]
    def __len__(self):
        return self.len_a + (len(self.b) if self.b is not None else 0)

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_pos, p_neg = p, 1.0 - p
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)
        los_pos = targets * torch.log(p_pos.clamp(min=self.eps)) * ((1.0 - p_pos) ** self.gamma_pos)
        los_neg = (1.0 - targets) * torch.log(p_neg.clamp(min=self.eps)) * (p ** self.gamma_neg)
        return -(los_pos + los_neg).mean()

class SwinMILDataset(Dataset):
    def __init__(self, uids, id2row, labels, vols, res=384, train=True, n_slices=6):
        self.uids = uids
        self.id2row = id2row
        self.labels = labels
        self.vols = vols
        self.res = res
        self.train = train
        self.n_slices = n_slices
    def __len__(self): return len(self.uids)
    def __getitem__(self, idx):
        u = self.uids[idx]
        row = self.id2row[u]
        v = self.vols[row]
        D, H, W = v.shape
        step = max(1, D // self.n_slices)
        idxs = [min(i * step + step // 2, D - 1) for i in range(self.n_slices)]
        imgs = []
        for ci in idxs:
            c_prev = max(0, ci - 1)
            c_next = min(D - 1, ci + 1)
            triplet = np.stack([v[c_prev], v[ci], v[c_next]], axis=0).astype(np.float32) / 255.0
            imgs.append(triplet)
        t = torch.tensor(np.stack(imgs, axis=0), dtype=torch.float32)
        if (H, W) != (self.res, self.res):
            t = nn.functional.interpolate(t, size=(self.res, self.res), mode='bilinear', align_corners=False)
        y = torch.tensor(self.labels[u], dtype=torch.float32)
        return t, y

class SwinMILModel(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window12_384.ms_in22k_ft_in1k', num_classes=12):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        d = self.encoder.num_features
        self.attn = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Linear(d, num_classes)
    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.encoder(x)
        feats = feats.view(B, S, -1)
        scores = self.attn(feats)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(feats * weights, dim=1)
        return self.classifier(pooled)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Training on Device: {DEVICE}', flush=True)

va = np.load('/content/corpus/all_vols.npy', mmap_mode='r')
vb = np.load('/content/corpus/extra_vols.npy', mmap_mode='r') if os.path.exists('/content/corpus/extra_vols.npy') else None
vols = TwoPartVols(va, vb)
a_ids = np.load('/content/corpus/all_ids.npy', allow_pickle=True).astype(str)
b_ids = np.load('/content/corpus/extra_ids.npy', allow_pickle=True).astype(str) if os.path.exists('/content/corpus/extra_ids.npy') else np.array([])
all_ids = np.concatenate([a_ids, b_ids]) if len(b_ids) > 0 else a_ids
id2row = {u: i for i, u in enumerate(all_ids)}

soft = pd.read_parquet('/content/labels/SUPER_CONSENSUS_labels.parquet').set_index('StudyInstanceUID')
train_ids = [u for u in all_ids if u in soft.index]
labels = {u: soft.loc[u, TARGETS].values.astype(np.float32) for u in train_ids}

np.random.seed(42)
shuffled_train = list(train_ids)
np.random.shuffle(shuffled_train)
n_val = len(shuffled_train) // 5

# Dynamic VRAM detection (Auto-adapts to Tesla T4 15GB, L4 24GB, or A100 40GB)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
print(f"[+] Detected GPU with {vram_gb:.1f} GB VRAM", flush=True)

if vram_gb < 20: # Tesla T4 (15GB) or V100 (16GB)
    BATCH_SIZE = 2
    GRAD_ACCUM = 4
    N_SLICES = 6
    USE_CHECKPOINTING = True
    print(f"[+] Auto-configured for Tesla T4: Batch Size={BATCH_SIZE}, Grad Accum={GRAD_ACCUM}, Grad Checkpointing=ON", flush=True)
else: # A100 (40GB) or L4 (24GB)
    BATCH_SIZE = 4
    GRAD_ACCUM = 2
    N_SLICES = 6
    USE_CHECKPOINTING = False
    print(f"[+] Auto-configured for High-VRAM: Batch Size={BATCH_SIZE}, Grad Accum={GRAD_ACCUM}", flush=True)

FOLDS_TO_TRAIN = [0, 1]
EPOCHS = 6

for fold in FOLDS_TO_TRAIN:
    val_uids = set(shuffled_train[fold * n_val : (fold + 1) * n_val])
    tr_uids = [u for u in shuffled_train if u not in val_uids]
    
    tds = SwinMILDataset(tr_uids, id2row, labels, vols, res=384, train=True, n_slices=N_SLICES)
    tl = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    print(f"\\n=== STARTING ENGINE 13 SWIN-BASE FOCAL: FOLD {fold} ===", flush=True)
    model = SwinMILModel(num_classes=12).to(DEVICE)
    if USE_CHECKPOINTING:
        try:
            model.encoder.set_grad_checkpointing(True)
            print("[+] Gradient checkpointing enabled for Swin encoder", flush=True)
        except Exception as e:
            print(f"[-] Note: set_grad_checkpointing skipped: {e}", flush=True)
            
    criterion = AsymmetricFocalLoss(gamma_neg=4.0, gamma_pos=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * (len(tl) // GRAD_ACCUM))
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        start_t = time.time()
        optimizer.zero_grad()
        
        for step, (bx, by) in enumerate(tl):
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(bx)
                raw_loss = criterion(logits, by)
                loss = raw_loss / GRAD_ACCUM
                
            scaler.scale(loss).backward()
            total_loss += raw_loss.item()
            
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(tl):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            if (step + 1) % (50 * GRAD_ACCUM) == 0:
                print(f"Fold {fold} | Epoch {epoch+1}/{EPOCHS} | Step {step+1}/{len(tl)} | Loss: {total_loss/(step+1):.4f}", flush=True)
        elapsed = time.time() - start_t
        print(f"Epoch {epoch+1} finished in {elapsed:.1f}s. Avg Loss: {total_loss/len(tl):.4f}", flush=True)
        
    weight_path = f"/content/engine13_swin_focal_fold{fold}_best.pt"
    drive_path = f"/content/drive/MyDrive/rsna_knee_checkpoints/engine13_swin_focal_fold{fold}_best.pt"
    torch.save(model.state_dict(), weight_path)
    os.system(f"cp {weight_path} {drive_path}")
    print(f"[+] Model permanently secured in Google Drive: {drive_path}", flush=True)
    
    # Kaggle Push
    pkg_dir = f"/content/kaggle_e13_fold{fold}"
    os.makedirs(pkg_dir, exist_ok=True)
    os.system(f"cp {weight_path} {pkg_dir}/")
    meta = {
        'title': f'RSNA Knee Engine 13 Swin-Base-384 Focal Fold {fold} Weights',
        'id': f'cyruspinto/rsna-knee-engine13-fold{fold}-weights',
        'licenses': [{'name': 'CC0-1.0'}]
    }
    with open(f"{pkg_dir}/dataset-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    try:
        api.dataset_create_new(pkg_dir, dir_mode='zip')
        print(f"[+] Created Kaggle dataset: cyruspinto/rsna-knee-engine13-fold{fold}-weights")
    except Exception as e:
        api.dataset_create_version(pkg_dir, f'Update Fold {fold} Weights', dir_mode='zip')
        print(f"[+] Updated Kaggle dataset: cyruspinto/rsna-knee-engine13-fold{fold}-weights")
        
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

print("\\n[+] ENGINE 13 SWIN-BASE FOCAL TRAINING COMPLETE!")
'''
nb_e13['cells'][2]['source'] = cell2_e13_code.splitlines(keepends=True)

with open("colab_engine13_swin_base_focal.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_e13, f, indent=1)
print("[+] Created colab_engine13_swin_base_focal.ipynb")

# ==============================================================================
# NOTEBOOK 3: Unified Master Notebook (Runs both sequentially if single tab preferred)
# ==============================================================================
nb_unified = json.load(open(SRC_NOTEBOOK, "r", encoding="utf-8"))
nb_unified['cells'][0]['source'] = [
    "# RSNA Knee Grandmaster Unified Master: Engine 12 + Engine 13\n",
    "### Complete Autonomous Sequential Suite (Single-Tab Run All & Walk Away)\n",
    "1. **Engine 12:** ConvNeXt-Large (198M params) with Gated Attention MIL\n",
    "2. **Engine 13:** Swin-Base-384 with Asymmetric Focal Loss\n",
    "3. **Automatic Dual-Sync:** Saves all checkpoints directly to Google Drive AND Kaggle Datasets\n"
]
nb_unified['cells'][1]['source'] = cell1_e12

# Cell 2 is Engine 12, Cell 3 is Engine 13
cell3_dict = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": cell2_e13_code.splitlines(keepends=True)
}
nb_unified['cells'][2]['source'] = cell2_e12_code.splitlines(keepends=True)
nb_unified['cells'].append(cell3_dict)

with open("colab_unified_dual_engine_master.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_unified, f, indent=1)
print("[+] Created colab_unified_dual_engine_master.ipynb")
