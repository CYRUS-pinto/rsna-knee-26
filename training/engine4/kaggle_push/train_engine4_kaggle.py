"""
RSNA Knee Abnormality Detection: Proprietary Engine 4 Training (Kaggle Dual T4)
Multi-Plane 2.5D ConvNeXt-Small with 12 Disentangled Gated Attention MIL Heads.

Key Breakthroughs over Engine 3:
1. True 5-Slot Anatomical Slicing: Samples from Sagittal T2, Sagittal T1, Coronal T2, Coronal T1, and Axial slots.
2. Correct Physical Plane IDs: [0, 1, 2, 3, 4] passed directly to learnable plane embeddings.
3. Memory-mapped (mmap) zero-copy reading from dreaddevelopment/knee-raptor-corpus (fastest possible I/O).
4. Supervised by continuous consensus targets (no noisy label binarization).
5. 12 independent Gated Attention MIL heads (one per pathology).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# Slot boundaries in the 96-slice Raptor volume:
# Slot 0: Sagittal T2 [0:26]
# Slot 1: Sagittal T1 [26:48]
# Slot 2: Coronal T2 [48:66]
# Slot 3: Coronal T1 [66:78]
# Slot 4: Axial [78:96]
SLOT_RANGES = [
    (0, 26, 0),   # (start, end, plane_id)
    (26, 48, 1),
    (48, 66, 2),
    (66, 78, 3),
    (78, 96, 4)
]
SLOT_PICKS = [5, 5, 5, 4, 5] # Total K = 24 slices per study

def exact_roc_auc(y_true, y_score):
    """Zero-dependency exact Wilcoxon-Mann-Whitney ROC-AUC."""
    y_true = np.asarray(y_true, bool)
    y_score = np.asarray(y_score, float)
    pos = y_score[y_true]
    neg = y_score[~y_true]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    all_scores = np.concatenate([pos, neg])
    ranks = pd.Series(all_scores).rank(method='average').to_numpy()
    sum_ranks_pos = ranks[:n_pos].sum()
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))

def _find(*names, roots=['/kaggle/input', '.']):
    for r in roots:
        if os.path.exists(r):
            for root, _, files in os.walk(r):
                for n in names:
                    if n in files:
                        return os.path.join(root, n)
    return None

class ConcatVols:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = len(a)
    def __getitem__(self, idx):
        return self.a[idx] if idx < self.n_a else self.b[idx - self.n_a]
    @property
    def shape(self):
        return (self.n_a + len(self.b),) + self.a.shape[1:]

class PerFindingAttentionMILHead(nn.Module):
    def __init__(self, in_features, num_classes=12, hidden_dim=256, num_planes=5, drop=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.plane_emb = nn.Embedding(num_planes, in_features)
        nn.init.trunc_normal_(self.plane_emb.weight, std=0.02)
        self.norm = nn.LayerNorm(in_features)
        
        self.attention_V = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh(), nn.Dropout(drop))
            for _ in range(num_classes)
        ])
        self.attention_U = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid(), nn.Dropout(drop))
            for _ in range(num_classes)
        ])
        self.attention_weights = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_classes)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(in_features, 1) for _ in range(num_classes)
        ])
        
    def forward(self, feats, plane_ids=None):
        B, K, D = feats.shape
        h = self.norm(feats)
        if plane_ids is not None:
            if plane_ids.dim() == 1:
                p_emb = self.plane_emb(plane_ids).unsqueeze(0).expand(B, -1, -1)
            else:
                p_emb = self.plane_emb(plane_ids)
            h = h + p_emb
            
        logits_list = []
        for j in range(self.num_classes):
            v = self.attention_V[j](h)
            u = self.attention_U[j](h)
            scores = self.attention_weights[j](v * u).squeeze(-1)
            attn = torch.softmax(scores, dim=-1)
            pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
            logit = self.classifiers[j](pooled)
            logits_list.append(logit)
        return torch.cat(logits_list, dim=-1)

class Engine4ConvNeXtMIL(nn.Module):
    def __init__(self, backbone_name='convnext_small', pretrained=True, in_chans=3, num_classes=12):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, 
            in_chans=in_chans, global_pool='avg'
        )
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            try:
                self.backbone.set_grad_checkpointing(True)
            except Exception:
                pass
        feat_dim = self.backbone.num_features
        self.head = PerFindingAttentionMILHead(in_features=feat_dim, num_classes=num_classes)
        
    def forward(self, x, plane_ids=None):
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        chunk_size = 16
        chunks = []
        for i in range(0, flat_x.size(0), chunk_size):
            chunks.append(self.backbone(flat_x[i:i + chunk_size]))
        feats = torch.cat(chunks, dim=0).view(B, K, -1)
        return self.head(feats, plane_ids=plane_ids)

class FastMultiPlaneDataset(Dataset):
    def __init__(self, uids, labels_dict, av, bv, ida, idb, am, bm, res=224, is_train=True):
        self.uids = uids
        self.labels_dict = labels_dict
        self.av, self.bv = av, bv
        self.ida, self.idb = ida, idb
        self.am, self.bm = am, bm
        self.res = res
        self.is_train = is_train
        self.vols = None
        self.masks = None
        self.id2row = None

        # Precompute target plane_ids vector of length sum(SLOT_PICKS) = 24
        p_list = []
        for (st, en, pid), n_pick in zip(SLOT_RANGES, SLOT_PICKS):
            p_list.extend([pid] * n_pick)
        self.plane_ids = torch.tensor(p_list, dtype=torch.long)

    def _init(self):
        if self.vols is None:
            va = np.load(self.av, mmap_mode='r')
            vb = np.load(self.bv, mmap_mode='r') if self.bv else np.zeros((0,) + va.shape[1:], dtype=va.dtype)
            ida = np.load(self.ida, allow_pickle=True).astype(str)
            idb = np.load(self.idb, allow_pickle=True).astype(str) if self.idb else np.array([])
            all_ids = np.concatenate([ida, idb])
            self.id2row = {u: i for i, u in enumerate(all_ids)}
            ma = np.load(self.am)
            mb = np.load(self.bm) if self.bm else np.zeros((0,) + ma.shape[1:], dtype=ma.dtype)
            self.masks = np.concatenate([ma, mb], axis=0)
            self.vols = ConcatVols(va, vb)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        self._init()
        uid = self.uids[idx]
        row = self.id2row.get(uid, 0)
        vol = self.vols[row]
        mask = self.masks[row]
        
        slabs = []
        # Sample across each slot separately to ensure TRUE 3-plane anatomical representation
        for (st, en, pid), n_pick in zip(SLOT_RANGES, SLOT_PICKS):
            slot_mask = mask[st:en]
            valid = np.where(slot_mask > 0)[0]
            if len(valid) == 0:
                valid = np.arange(en - st)
            
            # Keep safety margins for 2.5D triplets
            valid_indices = [v + st for v in valid if (v + st - 1) >= 0 and (v + st + 1) < vol.shape[0]]
            if not valid_indices:
                valid_indices = [max(1, min(st + (en - st) // 2, vol.shape[0] - 2))]
                
            if self.is_train:
                pool = valid_indices * (n_pick // len(valid_indices) + 1)
                np.random.shuffle(pool)
                picks = sorted(pool[:n_pick])
            else:
                steps = np.linspace(0, len(valid_indices) - 1, n_pick).round().astype(int)
                picks = sorted([valid_indices[i] for i in steps])
                
            for c in picks:
                # 2.5D triplet: (c-1, c, c+1)
                tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], axis=0).astype(np.float32) / 255.0
                t = torch.from_numpy(tri)
                if t.shape[-1] != self.res or t.shape[-2] != self.res:
                    t = F.interpolate(t.unsqueeze(0), size=(self.res, self.res), mode='bilinear', align_corners=False)[0]
                slabs.append(t)
                
        tensor_stack = torch.stack(slabs, dim=0) # (24, 3, res, res)
        y = torch.tensor(self.labels_dict.get(uid, np.zeros(12, dtype=np.float32)), dtype=torch.float32)
        return tensor_stack, self.plane_ids, y

def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"=== Starting Engine 4 Multi-Plane ConvNeXt Training on {dev} ===")
    
    # 1. Locate Datasets
    av = _find('all_vols.npy')
    bv = _find('all_vols_ext.npy')
    ida = _find('all_study_uids.npy')
    idb = _find('all_study_uids_ext.npy')
    am = _find('all_masks.npy')
    bm = _find('all_masks_ext.npy')
    labels_file = _find('SUPER_CONSENSUS_labels.parquet', 'SUPER_CONSENSUS_labels.csv')
    
    print(f"Files Located:\n  Volumes A: {av}\n  Labels: {labels_file}")
    assert av is not None and labels_file is not None, "Crucial dataset files missing!"
    
    df_labels = pd.read_parquet(labels_file) if labels_file.endswith('.parquet') else pd.read_csv(labels_file)
    uids = df_labels['StudyInstanceUID'].tolist()
    labels_dict = {row['StudyInstanceUID']: row[LABELS].values.astype(np.float32) for _, row in df_labels.iterrows()}
    
    # 5-Fold Group Split
    np.random.seed(42)
    shuffled_uids = list(uids)
    np.random.shuffle(shuffled_uids)
    val_count = len(shuffled_uids) // 5
    fold = 0
    val_uids = set(shuffled_uids[fold * val_count : (fold + 1) * val_count])
    train_uids = [u for u in shuffled_uids if u not in val_uids]
    val_uids_list = sorted(list(val_uids))
    print(f"Fold {fold}: Train = {len(train_uids)}, Val = {len(val_uids_list)}")
    
    train_ds = FastMultiPlaneDataset(train_uids, labels_dict, av, bv, ida, idb, am, bm, res=224, is_train=True)
    val_ds = FastMultiPlaneDataset(val_uids_list, labels_dict, av, bv, ida, idb, am, bm, res=224, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    model = Engine4ConvNeXtMIL(backbone_name='convnext_small', pretrained=True, in_chans=3, num_classes=12).to(dev)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    epochs = 6
    accum_steps = 8 # Effective batch size = 16
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if dev == 'cuda' else None
    
    best_macro_auc = 0.0
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()
        
        for step, (x, plane_ids, y) in enumerate(train_loader):
            x, plane_ids, y = x.to(dev, non_blocking=True), plane_ids.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(x, plane_ids=plane_ids)
                    loss = loss_fn(logits, y) / accum_steps
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model(x, plane_ids=plane_ids)
                loss = loss_fn(logits, y) / accum_steps
                loss.backward()
                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            train_loss += loss.item() * accum_steps
            
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, plane_ids, y in val_loader:
                x, plane_ids = x.to(dev, non_blocking=True), plane_ids.to(dev, non_blocking=True)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        logits = model(x, plane_ids=plane_ids)
                else:
                    logits = model(x, plane_ids=plane_ids)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(y.numpy())
                
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        class_aucs = {}
        for j, name in enumerate(LABELS):
            y_t = (val_targets[:, j] >= 0.5).astype(int)
            class_aucs[name] = exact_roc_auc(y_t, val_preds[:, j])
        macro_auc = float(np.mean(list(class_aucs.values())))
        ep_time = time.time() - t0
        print(f"Epoch {ep+1:02d}/{epochs:02d} [{ep_time:.1f}s] - Train Loss: {train_loss:.4f} | Val Macro-AUC: {macro_auc:.5f}")
        for name, a in class_aucs.items():
            print(f"   - {name:20s}: {a:.4f}")
            
        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            torch.save(model.state_dict(), f"/kaggle/working/engine4_convnext_fold{fold}_best.pth")
            df_oof = pd.DataFrame(val_preds, columns=LABELS)
            df_oof['StudyInstanceUID'] = val_uids_list
            df_oof.to_parquet(f"/kaggle/working/oof_engine4_fold{fold}.parquet", index=False)
            print(f">> Saved New Best Model (Macro-AUC: {macro_auc:.5f}) <<")
            
    print(f"=== ENGINE 4 TRAINING COMPLETE. Best Macro-AUC: {best_macro_auc:.5f} ===")

if __name__ == '__main__':
    main()
