"""
RSNA Knee Abnormality Detection - Multi-Plane Engine 3 (Kaggle Cloud Trainer)
100% Standalone Self-Contained Script
"""

import os
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

def exact_roc_auc(y_true, y_score):
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

def iterative_multilabel_split(y_binary, n_splits=5, seed=42):
    rng = np.random.RandomState(seed)
    n_samples, n_labels = y_binary.shape
    target_samples_per_fold = n_samples / n_splits
    target_pos_per_fold = y_binary.sum(axis=0) / n_splits
    fold_samples = np.zeros(n_splits, dtype=int)
    fold_pos = np.zeros((n_splits, n_labels), dtype=float)
    assignment = np.full(n_samples, -1, dtype=int)
    label_counts = y_binary.sum(axis=0)
    sample_rarity_scores = (y_binary * (1.0 / np.maximum(label_counts, 1))).sum(axis=1)
    jitter = rng.uniform(0, 1e-6, size=n_samples)
    sorted_indices = np.argsort(-(sample_rarity_scores + jitter))
    
    for idx in sorted_indices:
        labels = y_binary[idx]
        if labels.sum() == 0:
            best_fold = np.argmin(fold_samples)
        else:
            active_labels = np.where(labels > 0)[0]
            deficits = target_pos_per_fold[active_labels] - fold_pos[:, active_labels]
            fold_scores = deficits.sum(axis=1) - (fold_samples / target_samples_per_fold) * 0.1
            best_fold = np.argmax(fold_scores)
        assignment[idx] = best_fold
        fold_samples[best_fold] += 1
        fold_pos[best_fold] += labels
    return assignment

class MultiPlaneSliceTransformerHead(nn.Module):
    def __init__(self, F_dim=1024, n=12, num_heads=8, num_layers=2, max_slices=128, max_planes=8, drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(F_dim)
        self.intra_pos_emb = nn.Parameter(torch.zeros(1, max_slices, F_dim))
        nn.init.trunc_normal_(self.intra_pos_emb, std=0.02)
        self.plane_emb = nn.Embedding(max_planes, F_dim)
        nn.init.trunc_normal_(self.plane_emb.weight, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=F_dim, nhead=num_heads, dim_feedforward=F_dim * 2,
            dropout=drop, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, n))
        self.post_norm = nn.LayerNorm(F_dim)
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        
    def forward(self, feats, plane_ids=None):
        B, K, D = feats.shape
        h = self.norm(feats)
        pos = self.intra_pos_emb[:, :K, :]
        if plane_ids is not None:
            p_emb = self.plane_emb(plane_ids).unsqueeze(0) if plane_ids.dim() == 1 else self.plane_emb(plane_ids)
            h = h + pos + p_emb
        else:
            h = h + pos
        h_trans = self.transformer(h)
        a = torch.softmax(self.att(h_trans), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h_trans)
        pooled = self.post_norm(pooled)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

class MultiPlaneEngine3(nn.Module):
    def __init__(self, backbone_name='convnext_base.fb_in22k_ft_in1k_384', n=12, drop=0.1):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, in_chans=3, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.head = MultiPlaneSliceTransformerHead(F_dim=feat_dim, n=n, num_heads=8, num_layers=2, drop=drop)
        
    def forward(self, x, plane_ids=None):
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        chunk_size = 16
        chunks = [self.backbone(flat_x[i:i + chunk_size]) for i in range(0, flat_x.size(0), chunk_size)]
        feats = torch.cat(chunks, dim=0).view(B, K, -1)
        return self.head(feats, plane_ids=plane_ids)

class ConcatVols:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = len(a)
    def __getitem__(self, idx):
        return self.a[idx] if idx < self.n_a else self.b[idx - self.n_a]
    @property
    def shape(self):
        return (self.n_a + len(self.b),) + self.a.shape[1:]

def _find(*names, roots=['/kaggle/input', '/content', '.']):
    for r in roots:
        if os.path.exists(r):
            for root, _, files in os.walk(r):
                for n in names:
                    if n in files:
                        return os.path.join(root, n)
    return None

class KaggleKneeDataset(Dataset):
    def __init__(self, uids, labels_dict, av, bv, ida, idb, am, bm, k=16, res=384, is_train=True):
        self.uids = uids
        self.labels_dict = labels_dict
        self.av, self.bv = av, bv
        self.ida, self.idb = ida, idb
        self.am, self.bm = am, bm
        self.k = k
        self.res = res
        self.is_train = is_train
        self.vols = None
        self.masks = None
        self.id2row = None

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
        total_d = self.vols.shape[1]
        mask = self.masks[row]
        valid = np.where(mask > 0)[0]
        if len(valid) < 3:
            valid = np.arange(total_d)
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= 0 and c + 1 < total_d]
        if not cs:
            cs = [max(1, min((lo + hi) // 2, total_d - 2))]
        if self.is_train:
            pool = cs * (self.k // len(cs) + 1)
            np.random.shuffle(pool)
            chosen = sorted(pool[:self.k])
        else:
            step_indices = np.linspace(0, len(cs) - 1, self.k).round().astype(int)
            chosen = sorted([cs[i] for i in step_indices])
            
        vol = self.vols[row]
        slabs = []
        for c in chosen:
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], axis=0).astype(np.float32) / 255.0
            t = torch.from_numpy(tri)
            if t.shape[-1] != self.res or t.shape[-2] != self.res:
                t = F.interpolate(t.unsqueeze(0), size=(self.res, self.res), mode='bilinear', align_corners=False)[0]
            slabs.append(t)
        tensor_stack = torch.stack(slabs, dim=0)
        plane_ids = torch.zeros(self.k, dtype=torch.long)
        y = torch.tensor(self.labels_dict[uid], dtype=torch.float32)
        return tensor_stack, plane_ids, y

def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"=== Multi-Plane Engine 3 Training on {dev} ===")
    
    # Dynamic file discovery across all Kaggle input mounts
    av = _find('all_vols.npy')
    bv = _find('extra_vols.npy')
    ida = _find('all_ids.npy')
    idb = _find('extra_ids.npy')
    am = _find('all_masks.npy')
    bm = _find('extra_masks.npy')
    lab_path = _find('SUPER_CONSENSUS_labels.parquet', 'SUPER_CONSENSUS_labels.csv', 'labels_llm_soft.parquet')
    gold_path = _find('train.csv')
    
    print(f"Discovered Paths:")
    print(f"  all_vols: {av}")
    print(f"  extra_vols: {bv}")
    print(f"  labels: {lab_path}")
    print(f"  train.csv: {gold_path}")
    
    if not av or not lab_path or not gold_path:
        raise FileNotFoundError(f"Missing essential inputs: av={av}, lab={lab_path}, gold={gold_path}")
        
    df_super = pd.read_parquet(lab_path) if lab_path.endswith('.parquet') else pd.read_csv(lab_path)
    df_gold = pd.read_csv(gold_path)
    gold_studies = set(df_gold[df_gold[LABELS].notna().all(axis=1)]['StudyInstanceUID'].astype(str))
    df_super['StudyInstanceUID'] = df_super['StudyInstanceUID'].astype(str)
    df_super['is_gold'] = df_super['StudyInstanceUID'].isin(gold_studies)
    
    # Stratified Split (Seed 42)
    y_bin = (df_super[LABELS].values >= 0.5).astype(int)
    folds = np.zeros(len(df_super), dtype=int)
    gold_mask = df_super['is_gold'].values
    folds[gold_mask] = iterative_multilabel_split(y_bin[gold_mask], n_splits=5, seed=42)
    folds[~gold_mask] = iterative_multilabel_split(y_bin[~gold_mask], n_splits=5, seed=42)
    df_super['fold'] = folds
    
    target_fold = 0
    train_df = df_super[df_super['fold'] != target_fold].copy()
    val_df = df_super[df_super['fold'] == target_fold].copy()
    
    train_uids = train_df['StudyInstanceUID'].tolist()
    val_uids = val_df['StudyInstanceUID'].tolist()
    labels_dict = {row['StudyInstanceUID']: row[LABELS].values.astype(np.float32) for _, row in df_super.iterrows()}
    
    print(f"Fold {target_fold}: Train={len(train_uids)} | Val={len(val_uids)}")
    
    train_ds = KaggleKneeDataset(train_uids, labels_dict, av, bv, ida, idb, am, bm, k=16, is_train=True)
    val_ds = KaggleKneeDataset(val_uids, labels_dict, av, bv, ida, idb, am, bm, k=24, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    model = MultiPlaneEngine3(backbone_name='convnext_base.fb_in22k_ft_in1k_384').to(dev)
    loss_fn = nn.BCEWithLogitsLoss()
    
    bb_params = [p for n, p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': bb_params, 'lr': 3e-5},
        {'params': head_params, 'lr': 2e-4}
    ], weight_decay=0.01)
    
    epochs = 6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs, eta_min=1e-6)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    best_auc = 0.0
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for x, plane_ids, y in train_loader:
            x, plane_ids, y = x.to(dev, non_blocking=True), plane_ids.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(dev, dtype=autocast_dtype):
                logits = model(x, plane_ids=plane_ids)
                loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Eval
        model.eval()
        val_logits, val_targets = [], []
        with torch.no_grad():
            for x, plane_ids, y in val_loader:
                x, plane_ids = x.to(dev, non_blocking=True), plane_ids.to(dev, non_blocking=True)
                with torch.autocast(dev, dtype=autocast_dtype):
                    logits = model(x, plane_ids=plane_ids)
                val_logits.append(logits.float().cpu().numpy())
                val_targets.append(y.numpy())
                
        val_logits = np.concatenate(val_logits, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        aucs = {name: exact_roc_auc((val_targets[:, j] >= 0.5).astype(int), val_logits[:, j]) for j, name in enumerate(LABELS)}
        macro_auc = float(np.mean(list(aucs.values())))
        ep_time = time.time() - t0
        print(f"Epoch {ep+1:02d}/{epochs:02d} [{ep_time:.1f}s] - Loss: {train_loss:.4f} | Val OOF AUC: {macro_auc:.5f}")
        
        if macro_auc > best_auc:
            best_auc = macro_auc
            out_dir = '/kaggle/working' if os.path.exists('/kaggle') else 'weights'
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out_dir, f"engine3_multiplane_fold{target_fold}_best.pth"))
            print(f">> Saved Best Model: {macro_auc:.5f} <<")
            
    print(f"=== Complete. Best OOF Macro-AUC: {best_auc:.5f} ===")

if __name__ == '__main__':
    main()
