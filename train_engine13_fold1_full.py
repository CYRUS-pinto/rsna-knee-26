# 3. Write Engine 13: Swin-Base Fold 1 Master
with open('train_engine13_fold1.py', 'w', encoding='utf-8') as f:
    f.write('''"""
RSNA Knee Engine 13 — Swin-Base-384 Focal MIL (Fold 1 of 5)
5-Fold Stratified Cross-Validation + Official 58 Gold Radiologist Anchoring
"""
import os, sys, time, math, random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import timm

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class TwoPartVols:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + (b.shape[0] if b is not None else 0),) + tuple(a.shape[1:])
    def __len__(self): return self.shape[0]
    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]

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

class StudyWindowsDataset(Dataset):
    def __init__(self, ids, id2row, labels, vols, masks, res=384, k=16, train=True):
        self.ids = ids
        self.id2row = id2row
        self.labels = labels
        self.vols = vols
        self.masks = masks
        self.res = res
        self.k = k
        self.train = train

    def __len__(self): return len(self.ids)

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
            if random.random() < 0.5:
                x = torch.flip(x, dims=[-1])
            g = 1.0 + (random.random() - 0.5) * 0.15
            x = (x * g).clamp(0, 1)

        y = torch.tensor(self.labels[uid], dtype=torch.float32)
        return x, y

class SwinMILClassifier(nn.Module):
    def __init__(self, backbone, feat_dim=1024, num_targets=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.norm = nn.LayerNorm(feat_dim)
        self.att = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(256, num_targets)
        )
        self.clsW = nn.Parameter(torch.zeros(num_targets, feat_dim))
        self.clsb = nn.Parameter(torch.zeros(num_targets))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def encode(self, x):
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        chunk_size = 4
        feats = []
        for i in range(0, flat_x.size(0), chunk_size):
            f = self.backbone(flat_x[i:i + chunk_size])
            feats.append(f)
        feats = torch.cat(feats, dim=0)
        return feats.view(B, K, -1)

    def forward(self, x):
        feats = self.encode(x)
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

def main():
    print('=== Starting RSNA Knee Engine 13: Swin-Base Fold 1 (OOF Cross-Validation) ===', flush=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    va = np.load('/content/corpus/all_vols.npy', mmap_mode='r')
    vb = np.load('/content/corpus/extra_vols.npy', mmap_mode='r') if os.path.exists('/content/corpus/extra_vols.npy') else None
    vols = TwoPartVols(va, vb)

    ma = np.load('/content/corpus/all_masks.npy') if os.path.exists('/content/corpus/all_masks.npy') else np.ones(va.shape[:2], dtype=np.uint8)
    mb = np.load('/content/corpus/extra_masks.npy') if os.path.exists('/content/corpus/extra_masks.npy') else np.zeros((0,) + ma.shape[1:], dtype=ma.dtype)
    masks = np.concatenate([ma, mb], axis=0) if mb.shape[0] > 0 else ma

    a_ids = np.load('/content/corpus/all_ids.npy', allow_pickle=True).astype(str)
    b_ids = np.load('/content/corpus/extra_ids.npy', allow_pickle=True).astype(str) if os.path.exists('/content/corpus/extra_ids.npy') else np.array([])
    all_ids = np.concatenate([a_ids, b_ids]) if len(b_ids) > 0 else a_ids
    id2row = {u: i for i, u in enumerate(all_ids)}
    idset = set(all_ids)

    tr = pd.read_csv('/content/rsna_comp/train.csv')
    tr['StudyInstanceUID'] = tr['StudyInstanceUID'].astype(str)
    gold_df = tr[tr[TARGETS].notna().all(axis=1)].copy().set_index('StudyInstanceUID')
    gold_ids = [u for u in gold_df.index if u in idset]

    soft_path = '/content/labels/SUPER_CONSENSUS_labels.parquet'
    soft = pd.read_parquet(soft_path).set_index('StudyInstanceUID')

    train_pool = [u for u in all_ids if u in soft.index and u not in set(gold_ids)]
    labels = {u: soft.loc[u, TARGETS].values.astype(np.float32) for u in train_pool}
    for u in gold_ids:
        labels[u] = gold_df.loc[u, TARGETS].values.astype(np.float32)

    # 5-Fold Stratified Split (Seed 42)
    np.random.seed(42)
    shuffled_pool = list(train_pool)
    np.random.shuffle(shuffled_pool)
    n_val = len(shuffled_pool) // 5
    
    # Fold 1 Split:
    val_ids = shuffled_pool[1 * n_val : 2 * n_val]
    core_train_ids = [u for u in shuffled_pool if u not in set(val_ids)]
    print(f'[+] FOLD 1 SPLIT: {len(core_train_ids)} Train studies | {len(val_ids)} Out-of-Fold Val studies | {len(gold_ids)} Gold studies', flush=True)

    tds = StudyWindowsDataset(core_train_ids, id2row, labels, vols, masks, res=384, k=8, train=True)
    val_ds = StudyWindowsDataset(val_ids, id2row, labels, vols, masks, res=384, k=12, train=False)
    gds = StudyWindowsDataset(gold_ids, id2row, labels, vols, masks, res=384, k=12, train=False)

    tl = DataLoader(tds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    gl = DataLoader(gds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    bb = timm.create_model('swin_base_patch4_window12_384.ms_in22k_ft_in1k', pretrained=True, num_classes=0)
    try:
        bb.set_grad_checkpointing(True)
        print('[+] Gradient checkpointing enabled', flush=True)
    except Exception: pass
    
    model = SwinMILClassifier(bb, feat_dim=bb.num_features, num_targets=len(TARGETS)).to(device)

    head_params = [p for n, p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    bb_params = [p for n, p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]

    epochs = 8
    opt = torch.optim.AdamW([
        {'params': bb_params, 'lr': 3e-5},
        {'params': head_params, 'lr': 3e-4}
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-5, 3e-4], total_steps=len(tl) * epochs, pct_start=0.15)
    lossf = AsymmetricFocalLoss(gamma_neg=4.0, gamma_pos=1.0)

    best_oof_auc = 0.0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = lossf(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()

        model.eval()
        # 1. Out-of-Fold Validation
        P_val, Y_val = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                with torch.amp.autocast('cuda'):
                    out = model(vx.to(device))
                P_val.append(torch.sigmoid(out).float().cpu().numpy())
                Y_val.append(vy.numpy())
        P_val = np.concatenate(P_val, axis=0)
        Y_val = np.concatenate(Y_val, axis=0)
        oof_aucs = [roc_auc_score((Y_val[:, j] > 0.5).astype(int), P_val[:, j]) for j in range(12) if len(set((Y_val[:, j] > 0.5).astype(int))) > 1]
        macro_oof = float(np.mean(oof_aucs))

        # 2. Gold-58 Benchmark
        P_gold, Y_gold = [], []
        with torch.no_grad():
            for gx, gy in gl:
                with torch.amp.autocast('cuda'):
                    out_g = model(gx.to(device))
                P_gold.append(torch.sigmoid(out_g).float().cpu().numpy())
                Y_gold.append(gy.numpy())
        P_gold = np.concatenate(P_gold, axis=0)
        Y_gold = np.concatenate(Y_gold, axis=0)
        gold_aucs = [roc_auc_score(Y_gold[:, j], P_gold[:, j]) for j in range(12) if len(set(Y_gold[:, j].astype(int))) > 1]
        macro_gold = float(np.mean(gold_aucs))

        print(f'Epoch {ep+1:2d}/{epochs} | Loss: {tot/len(tl):.4f} | OOF Val AUC: {macro_oof:.4f} | Gold-58 AUC: {macro_gold:.4f} | {time.time()-t0:.0f}s', flush=True)

        if macro_oof > best_oof_auc or ep == epochs - 1:
            best_oof_auc = max(best_oof_auc, macro_oof)
            torch.save({
                'epoch': ep + 1,
                'model': model.state_dict(),
                'oof_auc': macro_oof,
                'gold_auc': macro_gold,
                'fold': 1,
                'arch': 'swin_base_focal_mil_384'
            }, '/content/engine13_swin_base_fold1_best.pt')
            print(f'  -> Saved Best Fold 1 Model (OOF AUC: {macro_oof:.4f}, Gold: {macro_gold:.4f})', flush=True)

    print(f'\n[+] FOLD 1 TRAINING FINISHED! Peak OOF Macro AUC: {best_oof_auc:.4f}', flush=True)

if __name__ == '__main__':
    main()
''')
print('Engine 13 Fold 1 training script generated!')
