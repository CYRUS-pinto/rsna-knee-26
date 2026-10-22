"""
RSNA Knee Abnormality Detection — Engine 7: DINOv2-S/14 2.5D Distilled Vision Master
Autonomous Kaggle GPU Training Pipeline with Dynamic Path Resolution
"""

import os
import sys
import time
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def _find(*names, roots=['/kaggle/input', '.']):
    for r in roots:
        if os.path.exists(r):
            for root, _, files in os.walk(r):
                for n in names:
                    if n in files:
                        return os.path.join(root, n)
    return None

def fast_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score)
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = np.empty(len(y_score), dtype=float)
    order = np.argsort(y_score)
    ranks[order] = np.arange(len(y_score))
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        tie_ranks = np.zeros(len(counts))
        np.add.at(tie_ranks, inv, ranks)
        tie_ranks = tie_ranks / counts
        ranks = tie_ranks[inv]
    rank_sum = np.sum(ranks[y_true])
    u = rank_sum - (n_pos * (n_pos - 1)) / 2.0
    return float(u / (n_pos * n_neg))

class TaskQueryAttentionHead(nn.Module):
    def __init__(self, embed_dim=384, num_targets=12, num_heads=6, drop=0.1):
        super().__init__()
        self.num_targets = num_targets
        self.embed_dim = embed_dim
        self.task_queries = nn.Parameter(torch.randn(num_targets, embed_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True
        )
        self.norm_tokens = nn.LayerNorm(embed_dim)
        self.norm_queries = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, token_seq, key_padding_mask=None):
        B = token_seq.shape[0]
        tokens = self.norm_tokens(token_seq)
        q = self.norm_queries(self.task_queries).unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(
            query=q, key=tokens, value=tokens,
            key_padding_mask=key_padding_mask, need_weights=False
        )
        fused = q + attn_out
        logits = self.mlp(fused).squeeze(-1)
        return logits

class DINOv2KneeClassifier(nn.Module):
    def __init__(self, backbone, embed_dim=384, num_targets=12, drop=0.1):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.num_targets = num_targets
        self.head = TaskQueryAttentionHead(
            embed_dim=embed_dim,
            num_targets=num_targets,
            drop=drop
        )

    def extract_features(self, x):
        B_times_K = x.shape[0]
        chunk_size = 16
        feats_list = []
        for i in range(0, B_times_K, chunk_size):
            chunk = x[i:i + chunk_size]
            if hasattr(self.backbone, 'forward_features'):
                out = self.backbone.forward_features(chunk)
                f = out[:, 0] if (hasattr(out, 'shape') and len(out.shape) == 3) else out
            else:
                f = self.backbone(chunk)
            feats_list.append(f)
        return torch.cat(feats_list, dim=0)

    def forward(self, x, mask=None):
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        feats = self.extract_features(flat_x)
        feats = feats.view(B, K, self.embed_dim)
        key_padding_mask = (mask == 0) if mask is not None else None
        logits = self.head(feats, key_padding_mask=key_padding_mask)
        return logits

class TwoPartVols:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + (b.shape[0] if b is not None else 0),) + tuple(a.shape[1:])
    def __len__(self): return self.shape[0]
    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]

class StudyWindows(Dataset):
    def __init__(self, ids, id2row, labels, vols, masks, res=224, k=16, train=True):
        self.ids = ids
        self.id2row = id2row
        self.labels = labels
        self.vols = vols
        self.masks = masks
        self.res = res
        self.k = k
        self.train = train

    def __len__(self):
        return len(self.ids)

    def _centers(self, valid, count, d_len):
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs:
            cs = [max(1, min((lo + hi) // 2, d_len - 2))]
        if self.train:
            reps = count // len(cs) + 1
            pool = (cs * reps)
            random.shuffle(pool)
            return sorted(pool[:count])
        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)
        return sorted([cs[i] for i in idx])

    def __getitem__(self, i):
        uid = self.ids[i]
        row = self.id2row[uid]
        vol = self.vols[row]
        m = self.masks[row]
        d_len = vol.shape[0]
        valid = np.where(m > 0)[0]
        if len(valid) < 3:
            valid = np.arange(min(3, d_len))
        cs = self._centers(valid, self.k, d_len)
        tiles = []
        for c in cs:
            c = max(1, min(c, d_len - 2))
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
            t = torch.from_numpy(np.ascontiguousarray(tri))
            if t.shape[-1] != self.res or t.shape[-2] != self.res:
                t = F.interpolate(t[None], size=(self.res, self.res), mode='bilinear', align_corners=False)[0]
            tiles.append(t)
        x = torch.stack(tiles, 0)
        if self.train:
            g = 1.0 + (random.random() - 0.5) * 0.15
            x = (x * g).clamp(0, 1)
        y = torch.tensor(self.labels[uid], dtype=torch.float32)
        return x, y

def main():
    print("=== Launching RSNA Knee Engine 7 DINOv2-S/14 Distilled Vision Master ===", flush=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Dynamic Path Resolution
    av = _find('all_vols.npy')
    bv = _find('all_vols_ext.npy', 'extra_vols.npy')
    ida = _find('all_ids.npy', 'all_study_uids.npy')
    idb = _find('all_ids_ext.npy', 'extra_ids.npy', 'all_study_uids_ext.npy')
    am = _find('all_masks.npy')
    bm = _find('all_masks_ext.npy', 'extra_masks.npy')
    labels_file = _find('SUPER_CONSENSUS_labels.parquet', 'SUPER_CONSENSUS_labels.csv')
    comp_train_file = _find('train.csv')

    print(f"Dataset Files Found:\n  Volumes: {av}\n  Extra Volumes: {bv}\n  IDs: {ida}\n  Labels: {labels_file}\n  Train CSV: {comp_train_file}", flush=True)
    assert av is not None and ida is not None and labels_file is not None and comp_train_file is not None, "Crucial dataset files missing!"

    # Load corpus
    va = np.load(av, mmap_mode='r')
    vb = np.load(bv, mmap_mode='r') if bv else None
    vols = TwoPartVols(va, vb)
    
    ma = np.load(am) if am else np.ones(va.shape[:2], dtype=np.uint8)
    mb = np.load(bm) if bm else np.zeros((0,) + ma.shape[1:], dtype=ma.dtype)
    masks = np.concatenate([ma, mb], axis=0) if bm else ma

    a_ids = np.load(ida, allow_pickle=True).astype(str)
    b_ids = np.load(idb, allow_pickle=True).astype(str) if idb else np.array([])
    all_ids = np.concatenate([a_ids, b_ids]) if len(b_ids) > 0 else a_ids
    
    id2row = {u: i for i, u in enumerate(all_ids)}
    idset = set(all_ids)
    print(f"Loaded corpus: {len(all_ids)} total volumetric examinations.", flush=True)

    # Load labels and train.csv
    tr = pd.read_csv(comp_train_file)
    tr['StudyInstanceUID'] = tr['StudyInstanceUID'].astype(str)
    gold_df = tr[tr[TARGETS].notna().all(axis=1)].copy().set_index('StudyInstanceUID')
    gold_ids = [u for u in gold_df.index if u in idset]
    print(f"Anchored on {len(gold_ids)} official radiologist gold studies.", flush=True)

    soft = pd.read_parquet(labels_file) if str(labels_file).endswith('.parquet') else pd.read_csv(labels_file)
    soft['StudyInstanceUID'] = soft['StudyInstanceUID'].astype(str)
    soft = soft.set_index('StudyInstanceUID')

    train_ids = [u for u in all_ids if u in soft.index and u not in set(gold_ids)]
    labels = {u: soft.loc[u, TARGETS].values.astype(np.float32) for u in train_ids}
    for u in gold_ids:
        labels[u] = gold_df.loc[u, TARGETS].values.astype(np.float32)

    # 5-fold split (fold 0 as validation)
    np.random.seed(42)
    shuffled_train = list(train_ids)
    np.random.shuffle(shuffled_train)
    n_val = len(shuffled_train) // 5
    val_ids = shuffled_train[:n_val]
    core_train_ids = shuffled_train[n_val:]
    print(f"Training split: {len(core_train_ids)} train | {len(val_ids)} val | {len(gold_ids)} gold anchor.", flush=True)

    res = 224
    k = 16
    bs = 8
    accum = 2
    epochs = 8

    tds = StudyWindows(core_train_ids, id2row, labels, vols, masks, res=res, k=k, train=True)
    vds = StudyWindows(val_ids, id2row, labels, vols, masks, res=res, k=k, train=False)
    gds = StudyWindows(gold_ids, id2row, labels, vols, masks, res=res, k=k, train=False)

    tl = DataLoader(tds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(vds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    gl = DataLoader(gds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize DINOv2 Backbone
    print("Loading pretrained DINOv2-S/14 via torch.hub / timm...", flush=True)
    try:
        import timm
        backbone = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=224, dynamic_img_size=True)
        embed_dim = backbone.num_features
    except Exception as e:
        print(f"timm failed: {e}. Trying torch.hub...", flush=True)
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        embed_dim = 384

    model = DINOv2KneeClassifier(backbone=backbone, embed_dim=embed_dim, num_targets=len(TARGETS)).to(device)

    head_params = [p for n, p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    bb_params = [p for n, p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]

    opt = torch.optim.AdamW([
        {'params': bb_params, 'lr': 2e-5},
        {'params': head_params, 'lr': 2e-4}
    ], weight_decay=0.01)

    total_steps = math.ceil(len(tl) / accum) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=1e-6)
    lossf = nn.BCEWithLogitsLoss()

    best_gold_auc = 0.0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        tot = 0.0
        opt.zero_grad()
        for step, (x, y) in enumerate(tl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = lossf(logits, y) / accum
            loss.backward()

            if (step + 1) % accum == 0 or (step + 1) == len(tl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                sched.step()

            tot += loss.item() * accum

        # Evaluate on Gold-58 anchor
        model.eval()
        gold_preds, gold_trues = [], []
        with torch.no_grad():
            for gx, gy in gl:
                with torch.amp.autocast('cuda'):
                    g_logits = model(gx.to(device))
                gold_preds.append(torch.sigmoid(g_logits).float().cpu().numpy())
                gold_trues.append(gy.numpy())

        P = np.concatenate(gold_preds, axis=0)
        Y = np.concatenate(gold_trues, axis=0)
        gold_aucs = [fast_roc_auc(Y[:, j], P[:, j]) for j in range(len(TARGETS))]
        macro_gold = float(np.nanmean(gold_aucs))

        avg_loss = tot / len(tl)
        elapsed = time.time() - t0
        print(f"Epoch {ep+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Gold-58 Macro AUC: {macro_gold:.4f} (Best: {max(best_gold_auc, macro_gold):.4f}) | Elapsed: {elapsed:.0f}s", flush=True)

        if macro_gold > best_gold_auc:
            best_gold_auc = macro_gold
            torch.save({
                'epoch': ep + 1,
                'state_dict': model.state_dict(),
                'gold_auc': macro_gold,
                'targets': TARGETS
            }, "/kaggle/working/dinov2_f0_best.pt")
            print(f"  -> Saved new best checkpoint with Gold AUC: {macro_gold:.4f}", flush=True)

    print(f"\nTraining Complete. Peak Gold-58 Macro-AUC: {best_gold_auc:.4f}", flush=True)

if __name__ == '__main__':
    main()
