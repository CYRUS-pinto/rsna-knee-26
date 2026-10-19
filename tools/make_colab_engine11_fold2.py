"""
Script to generate Colab Autonomous Master Notebook for:
Engine 11 ConvNeXt-Base (88M params) - Fold 2 Training
Target: Google Colab A100 GPU
Loss: Asymmetric Focal Loss (ASL) + Mixed Precision (AMP)
Resolution: 384x384px (8 slices x 3 = 24 2.5D channels)
"""

import json
from pathlib import Path

nb_src = json.load(open("colab_engine9_fold1_master.ipynb", "r", encoding="utf-8"))

# Update Title
nb_src["cells"][0]["source"] = [
    "# Autonomous Engine 11 Fold 2: ConvNeXt-Base 384px High-Capacity MIL (A100 Master)\n",
    "\n",
    "**Objective:** Complete Fold 2 of the 88M-parameter Apex ConvNeXt-Base backbone to achieve a multi-fold sovereign ensemble.\n",
    "- **Architecture:** `convnext_base.fb_in22k_ft_in1k_384` (88M parameters, feat_dim=1024)\n",
    "- **Resolution:** 384x384px, 2.5D Triplet Windowing\n",
    "- **Folds:** Fold 2 (val_ids = [2*n_val : 3*n_val])\n",
    "- **Training Time:** ~35 minutes on Colab A100 (~1.2 compute units)\n",
    "- **Target Gold AUC:** >= 0.8800"
]

# Update Training Script Cell (Cell 3)
train_script_content = """with open('train_engine11_fold2.py', 'w') as f:
    f.write('''import os, sys, math, time, random, gc
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class TwoPartVols:
    def __init__(self, va, vb=None):
        self.va = va
        self.vb = vb
        self.na = len(va) if va is not None else 0
    def __getitem__(self, idx):
        if idx < self.na:
            return self.va[idx]
        return self.vb[idx - self.na]

class MIL384Dataset(Dataset):
    def __init__(self, uids, id2row, labels, vols, res=384, train=True, n_slices=8):
        self.uids = uids
        self.id2row = id2row
        self.labels = labels
        self.vols = vols
        self.res = res
        self.train = train
        self.n_slices = n_slices

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        row_idx = self.id2row[uid]
        vol = self.vols[row_idx]  # (D, H, W)
        D, H, W = vol.shape

        if D < 3:
            vol = np.pad(vol, ((0, max(0, 3 - D)), (0, 0), (0, 0)), mode='edge')
            D = vol.shape[0]

        step = max(1, (D - 2) // self.n_slices)
        centers = [min(D - 2, 1 + i * step) for i in range(self.n_slices)]

        triplets = []
        for c in centers:
            trip = vol[c-1:c+2].astype(np.float32) / 255.0
            triplets.append(trip)

        stack = np.stack(triplets, axis=0)  # (n_slices, 3, H, W)
        t = torch.from_numpy(np.ascontiguousarray(stack))

        if t.shape[-1] != self.res or t.shape[-2] != self.res:
            t = F.interpolate(t, size=(self.res, self.res), mode='bilinear', align_corners=False)

        if self.train:
            if random.random() < 0.5:
                t = torch.flip(t, dims=[-1])
            gain = 1.0 + (random.random() - 0.5) * 0.12
            t = (t * gain).clamp(0, 1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t = (t - mean) / std

        y = torch.tensor(self.labels[uid], dtype=torch.float32)
        return t, y

class Engine11MIL(nn.Module):
    def __init__(self, feat_dim=1024, model_name='convnext_base.fb_in22k_ft_in1k_384'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.norm = nn.LayerNorm(feat_dim)
        self.att = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, 12)
        )
        self.clsW = nn.Parameter(torch.empty(12, feat_dim))
        self.clsb = nn.Parameter(torch.empty(12))
        nn.init.xavier_uniform_(self.clsW)
        nn.init.zeros_(self.clsb)

    def forward(self, x):
        # x: (B, S, 3, 384, 384)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        h = self.backbone(x)  # (B*S, 1024)
        h = self.norm(h)
        h = h.view(B, S, -1)

        att_logits = self.att(h)  # (B, S, 12)
        alpha = F.softmax(att_logits, dim=1)  # (B, S, 12)

        # z: (B, 12, 1024)
        z = torch.einsum('bsk,bsd->bkd', alpha, h)
        logits = (z * self.clsW.unsqueeze(0)).sum(dim=-1) + self.clsb.unsqueeze(0)  # (B, 12)
        return logits

def main():
    print('=== Starting RSNA Knee Engine 11 Fold 2: ConvNeXt-Base 384px 2.5D MIL ===', flush=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    # Load Corpus
    va = np.load('/content/corpus/all_vols.npy', mmap_mode='r')
    vb = np.load('/content/corpus/extra_vols.npy', mmap_mode='r') if os.path.exists('/content/corpus/extra_vols.npy') else None
    vols = TwoPartVols(va, vb)

    a_ids = np.load('/content/corpus/all_ids.npy', allow_pickle=True).astype(str)
    b_ids = np.load('/content/corpus/extra_ids.npy', allow_pickle=True).astype(str) if os.path.exists('/content/corpus/extra_ids.npy') else np.array([])
    all_ids = np.concatenate([a_ids, b_ids]) if len(b_ids) > 0 else a_ids

    id2row = {u: i for i, u in enumerate(all_ids)}
    idset = set(all_ids)
    print(f'Loaded corpus: {len(all_ids)} examinations.', flush=True)

    # Load Gold Anchor & Continuous Soft Labels
    tr = pd.read_csv('/content/rsna_comp/train.csv')
    tr['StudyInstanceUID'] = tr['StudyInstanceUID'].astype(str)
    gold_df = tr[tr[TARGETS].notna().all(axis=1)].copy().set_index('StudyInstanceUID')
    gold_ids = [u for u in gold_df.index if u in idset]
    print(f'Anchored on {len(gold_ids)} official radiologist gold studies.', flush=True)

    soft_path = '/content/labels/SUPER_CONSENSUS_labels.parquet'
    soft = pd.read_parquet(soft_path).set_index('StudyInstanceUID')

    train_ids = [u for u in all_ids if u in soft.index and u not in set(gold_ids)]
    labels = {u: soft.loc[u, TARGETS].values.astype(np.float32) for u in train_ids}
    for u in gold_ids:
        labels[u] = gold_df.loc[u, TARGETS].values.astype(np.float32)

    # 5-fold split: FOLD 2 (disjoint from Fold 0 & Fold 1)
    np.random.seed(42)
    shuffled_train = list(train_ids)
    np.random.shuffle(shuffled_train)
    n_val = len(shuffled_train) // 5
    val_ids = shuffled_train[2*n_val:3*n_val]
    core_train_ids = shuffled_train[:2*n_val] + shuffled_train[3*n_val:]
    print(f'Fold 2 Split: {len(core_train_ids)} train, {len(val_ids)} validation.', flush=True)

    tds = MIL384Dataset(core_train_ids, id2row, labels, vols, res=384, train=True, n_slices=8)
    vds = MIL384Dataset(val_ids, id2row, labels, vols, res=384, train=False, n_slices=8)
    gds = MIL384Dataset(gold_ids, id2row, labels, vols, res=384, train=False, n_slices=8)

    # Batch size 8 on A100 for Base (8 * 8 slices = 64 forward passes per batch)
    tl = DataLoader(tds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(vds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    gl = DataLoader(gds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    model = Engine11MIL(feat_dim=1024, model_name='convnext_base.fb_in22k_ft_in1k_384').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    best_gold_auc = 0.0
    for epoch in range(1, 9):
        model.train()
        total_loss, steps = 0.0, 0
        t0 = time.time()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            steps += 1
        scheduler.step()
        train_loss = total_loss / max(1, steps)

        # Eval on Gold Radiologist Ground Truth
        model.eval()
        g_preds, g_trues = [], []
        with torch.no_grad():
            for x, y in gl:
                x = x.to(device)
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    probs = torch.sigmoid(logits).cpu().numpy()
                g_preds.append(probs)
                g_trues.append(y.numpy())
        g_preds = np.concatenate(g_preds, axis=0)
        g_trues = np.concatenate(g_trues, axis=0)

        aucs = []
        for j in range(12):
            if len(np.unique(g_trues[:, j])) > 1:
                aucs.append(roc_auc_score(g_trues[:, j], g_preds[:, j]))
        gold_macro = float(np.mean(aucs)) if len(aucs) > 0 else 0.0
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/8 | Loss: {train_loss:.4f} | Gold AUC: {gold_macro:.4f} | {elapsed:.0f}s", flush=True)

        if gold_macro > best_gold_auc:
            best_gold_auc = gold_macro
            ckpt = {
                'epoch': epoch,
                'gold_auc': gold_macro,
                'model': model.state_dict()
            }
            torch.save(ckpt, 'engine11_convnext_base_384_fold2_best.pt')
            print(f"--> Saved NEW BEST Engine 11 Fold 2 Model: {gold_macro:.4f}", flush=True)

    print(f"Training Complete. Peak Gold Macro AUC: {best_gold_auc:.4f}", flush=True)

if __name__ == '__main__':
    main()
''')
"""

nb_src["cells"][3]["source"] = train_script_content.splitlines(keepends=True)
nb_src["cells"][4]["source"] = ["!python train_engine11_fold2.py\n"]

# Update Kaggle Dataset upload cell (Cell 5)
nb_src["cells"][5]["source"] = [
    "# 5. Package & Publish Fold 2 Weights to Kaggle Datasets\n",
    "import os, json\n",
    "\n",
    "os.makedirs('/content/e11_dataset', exist_ok=True)\n",
    "!cp engine11_convnext_base_384_fold2_best.pt /content/e11_dataset/\n",
    "\n",
    "meta = {\n",
    '    "title": "RSNA Knee Engine 11 Fold 2 ConvNeXt-Base Weights",\n',
    '    "id": "cyruspinto/rsna-knee-engine11-fold2-weights",\n',
    '    "licenses": [{"name": "CC0-1.0"}]\n',
    "}\n",
    "with open('/content/e11_dataset/dataset-metadata.json', 'w') as f:\n",
    "    json.dump(meta, f, indent=2)\n",
    "\n",
    "!kaggle datasets create -p /content/e11_dataset -r zip\n"
]

out_nb = Path("colab_engine11_fold2_master.ipynb")
json.dump(nb_src, open(out_nb, "w", encoding="utf-8"), indent=1)
print(f"SUCCESS: Generated {out_nb}")
