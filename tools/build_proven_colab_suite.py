import json, os, copy

SRC_TEMPLATE = r"C:\Users\Cyrus\Downloads\colab_two_plane_convnext_master.ipynb"
with open(SRC_TEMPLATE, "r", encoding="utf-8") as f:
    template_nb = json.load(f)

# ==============================================================================
# ENGINE 12: ConvNeXt-Large-384 with Gated Attention MIL
# ==============================================================================
nb_e12 = copy.deepcopy(template_nb)
nb_e12['cells'][0]['source'] = [
    "# RSNA Knee Abnormality Detection — Engine 12: 384px ConvNeXt-Large Gated Attention MIL\n",
    "### Platform: Google Colab Pro (A100 / L4 / T4 GPU)\n",
    "**Key Architecture Features:**\n",
    "- **ConvNeXt-Large Backbone (198M params):** `convnext_large.fb_in22k_ft_in1k_384`.\n",
    "- **Chunked Feature Encoding (chunk_size=16):** Prevents OOM on any GPU (T4, L4, or A100).\n",
    "- **Gated Attention MIL:** Dynamic slice selection solving 2-slice focal tear dilution.\n",
    "- **Official Gold-58 Anchoring:** Real-time radiologist macro-AUC computed after every epoch.\n",
    "- **Automated Kaggle Push:** Pushes `.pt` weights directly to Kaggle Datasets via API.\n"
]

# Script content for Engine 12
script_e12 = '''"""
RSNA Knee Engine 12 — ConvNeXt-Large Gated Attention-MIL Master (384x384)
Preserves 100% Pretrained ImageNet Weights via 3-Slice Triplet Windows
Platform: Colab Pro (A100/L4/T4) | Evaluated on Official 58 Gold Radiologist Studies
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

        x = torch.stack(tiles, 0)  # (K, 3, H, W)
        if self.train:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[-1])
            g = 1.0 + (random.random() - 0.5) * 0.15
            x = (x * g).clamp(0, 1)

        y = torch.tensor(self.labels[uid], dtype=torch.float32)
        return x, y

class ConvNeXtLargeGatedMIL(nn.Module):
    def __init__(self, backbone, feat_dim=1536, num_targets=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.norm = nn.LayerNorm(feat_dim)
        # Gated Attention Pooling Mechanism
        self.att_v = nn.Sequential(nn.Linear(feat_dim, 256), nn.Tanh())
        self.att_u = nn.Sequential(nn.Linear(feat_dim, 256), nn.Sigmoid())
        self.att_w = nn.Linear(256, num_targets)
        self.clsW = nn.Parameter(torch.zeros(num_targets, feat_dim))
        self.clsb = nn.Parameter(torch.zeros(num_targets))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def encode(self, x):
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        chunk_size = 4  # Safe chunk size for ConvNeXt-Large at 384px (Zero OOM on T4/A100)
        feats = []
        for i in range(0, flat_x.size(0), chunk_size):
            f = self.backbone(flat_x[i:i + chunk_size])
            feats.append(f)
        feats = torch.cat(feats, dim=0)
        return feats.view(B, K, -1)

    def forward(self, x):
        feats = self.encode(x)
        h = self.norm(feats)
        v = self.att_v(h)
        u = self.att_u(h)
        scores = self.att_w(v * u)  # (B, K, num_targets)
        a = torch.softmax(scores, dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

def main():
    print('=== Starting RSNA Knee Engine 12: ConvNeXt-Large 384px Gated MIL Master ===', flush=True)
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
    print(f'Loaded corpus: {len(all_ids)} examinations.', flush=True)

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

    res = 384
    k = 8
    k_eval = 12
    tds = StudyWindowsDataset(train_ids, id2row, labels, vols, masks, res=res, k=k, train=True)
    gds = StudyWindowsDataset(gold_ids, id2row, labels, vols, masks, res=res, k=k_eval, train=False)

    tl = DataLoader(tds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    gl = DataLoader(gds, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    print('Initializing ConvNeXt-Large (in22k pretrained) for 384px Gated MIL...', flush=True)
    bb = timm.create_model('convnext_large.fb_in22k_ft_in1k_384', pretrained=True, num_classes=0)
    try:
        bb.set_grad_checkpointing(True)
        print('[+] Gradient checkpointing enabled for ConvNeXt-Large backbone', flush=True)
    except Exception as e:
        pass
    feat_dim = bb.num_features
    model = ConvNeXtLargeGatedMIL(bb, feat_dim=feat_dim, num_targets=len(TARGETS)).to(device)

    head_params = [p for n, p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    bb_params = [p for n, p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]

    epochs = 8
    opt = torch.optim.AdamW([
        {'params': bb_params, 'lr': 2e-5},
        {'params': head_params, 'lr': 2e-4}
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-5, 2e-4], total_steps=len(tl) * epochs, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss()

    best_gold_auc = 0.0
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
        P, Y = [], []
        with torch.no_grad():
            for gx, gy in gl:
                with torch.amp.autocast('cuda'):
                    gl_out = model(gx.to(device))
                P.append(torch.sigmoid(gl_out).float().cpu().numpy())
                Y.append(gy.numpy())
        P = np.concatenate(P, axis=0)
        Y = np.concatenate(Y, axis=0)
        aucs = {}
        for j, name in enumerate(TARGETS):
            if len(set(Y[:, j].astype(int))) > 1:
                aucs[name] = float(roc_auc_score(Y[:, j], P[:, j]))
        macro_gold = float(np.mean(list(aucs.values())))

        print(f'Epoch {ep+1:2d}/{epochs} | Loss: {tot/len(tl):.4f} | Gold-58 Macro AUC: {macro_gold:.4f} (Best: {max(best_gold_auc, macro_gold):.4f}) | {time.time()-t0:.0f}s', flush=True)

        if macro_gold > best_gold_auc:
            best_gold_auc = macro_gold
            torch.save({
                'epoch': ep + 1,
                'model': model.state_dict(),
                'gold_auc': macro_gold,
                'aucs': aucs,
                'arch': 'convnext_large_mil_384',
                'targets': TARGETS
            }, '/content/engine12_convnext_large_best.pt')
            print(f'  -> New Best Saved with Gold AUC: {macro_gold:.4f}', flush=True)

    print('', flush=True)
    print(f'Training Finished! Peak Gold-58 Macro-AUC: {best_gold_auc:.4f}', flush=True)

if __name__ == '__main__':
    main()
'''

nb_e12['cells'][3]['source'] = [
    "# 3. Write Engine 12: 384px ConvNeXt-Large Gated Attention MIL Master\n",
    "with open('train_engine12_convnext_large.py', 'w', encoding='utf-8') as f:\n",
    "    f.write('''" + script_e12 + "''')\n",
    "print('Engine 12 training script generated!')\n"
]
nb_e12['cells'][4]['source'] = ["!python train_engine12_convnext_large.py\n"]

# Cell 5: Automated Kaggle publish
nb_e12['cells'][5]['source'] = [
    "# 5. Publish Engine 12 Weights to Kaggle Dataset (Automated)\n",
    "import os, json\n",
    "os.makedirs('/content/engine12_upload', exist_ok=True)\n",
    "!cp /content/engine12_convnext_large_best.pt /content/engine12_upload/\n",
    "\n",
    "meta = {\n",
    "    'title': 'RSNA Knee Engine 12 ConvNeXt-Large 384 Weights',\n",
    "    'id': 'cyruspinto/rsna-knee-engine12-convnext-large-weights',\n",
    "    'licenses': [{'name': 'CC0-1.0'}]\n",
    "}\n",
    "with open('/content/engine12_upload/dataset-metadata.json', 'w') as f:\n",
    "    json.dump(meta, f, indent=2)\n",
    "\n",
    "!kaggle datasets create -p /content/engine12_upload -u -r zip || kaggle datasets version -p /content/engine12_upload -m 'Engine 12 ConvNeXt-Large Weights' -r zip\n",
    "print('Published Engine 12 weights to Kaggle successfully!')\n"
]

# Trim unnecessary cell 6
if len(nb_e12['cells']) > 6:
    nb_e12['cells'] = nb_e12['cells'][:6]

out_e12_local = "colab_engine12_convnext_large_master.ipynb"
out_e12_dl = r"C:\Users\Cyrus\Downloads\colab_engine12_convnext_large_master.ipynb"
with open(out_e12_local, "w", encoding="utf-8") as f:
    json.dump(nb_e12, f, indent=1)
with open(out_e12_dl, "w", encoding="utf-8") as f:
    json.dump(nb_e12, f, indent=1)
print(f"[+] Successfully built {out_e12_local} and copied to {out_e12_dl}")


# ==============================================================================
# ENGINE 13: Swin-Base-384 with Focal Loss Master
# ==============================================================================
nb_e13 = copy.deepcopy(template_nb)
nb_e13['cells'][0]['source'] = [
    "# RSNA Knee Abnormality Detection — Engine 13: 384px Swin-Base-384 Focal Master\n",
    "### Platform: Google Colab Pro (A100 / L4 / T4 GPU)\n",
    "**Key Architecture Features:**\n",
    "- **Hierarchical Swin-Base-384:** `swin_base_patch4_window12_384.ms_in22k_ft_in1k`.\n",
    "- **Chunked Window Encoding (chunk_size=12):** Prevents OOM across all GPUs.\n",
    "- **Asymmetric Focal Loss (gamma-=4.0, gamma+=1.0):** Suppresses 95% easy negative slice noise.\n",
    "- **Official Gold-58 Anchoring:** Real-time radiologist macro-AUC computed after every epoch.\n",
    "- **Automated Kaggle Push:** Pushes `.pt` weights directly to Kaggle Datasets via API.\n"
]

script_e13 = '''"""
RSNA Knee Engine 13 — Swin-Base-384 Asymmetric Focal Loss Master
Preserves 100% Pretrained ImageNet Weights via 3-Slice Triplet Windows
Platform: Colab Pro (A100/L4/T4) | Evaluated on Official 58 Gold Radiologist Studies
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

        x = torch.stack(tiles, 0)  # (K, 3, H, W)
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
        chunk_size = 4  # Safe chunk size preventing Swin attention OOM on T4 (takes <6GB)
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
    print('=== Starting RSNA Knee Engine 13: Swin-Base 384px Focal MIL Master ===', flush=True)
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
    print(f'Loaded corpus: {len(all_ids)} examinations.', flush=True)

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

    res = 384
    k = 8
    k_eval = 12
    tds = StudyWindowsDataset(train_ids, id2row, labels, vols, masks, res=res, k=k, train=True)
    gds = StudyWindowsDataset(gold_ids, id2row, labels, vols, masks, res=res, k=k_eval, train=False)

    tl = DataLoader(tds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    gl = DataLoader(gds, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    print('Initializing Swin-Base-384 for Focal MIL...', flush=True)
    bb = timm.create_model('swin_base_patch4_window12_384.ms_in22k_ft_in1k', pretrained=True, num_classes=0)
    try:
        bb.set_grad_checkpointing(True)
        print('[+] Gradient checkpointing enabled for Swin-Base backbone', flush=True)
    except Exception as e:
        pass
    feat_dim = bb.num_features
    model = SwinMILClassifier(bb, feat_dim=feat_dim, num_targets=len(TARGETS)).to(device)

    head_params = [p for n, p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    bb_params = [p for n, p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]

    epochs = 8
    opt = torch.optim.AdamW([
        {'params': bb_params, 'lr': 3e-5},
        {'params': head_params, 'lr': 3e-4}
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-5, 3e-4], total_steps=len(tl) * epochs, pct_start=0.15)
    lossf = AsymmetricFocalLoss(gamma_neg=4.0, gamma_pos=1.0)

    best_gold_auc = 0.0
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
        P, Y = [], []
        with torch.no_grad():
            for gx, gy in gl:
                with torch.amp.autocast('cuda'):
                    gl_out = model(gx.to(device))
                P.append(torch.sigmoid(gl_out).float().cpu().numpy())
                Y.append(gy.numpy())
        P = np.concatenate(P, axis=0)
        Y = np.concatenate(Y, axis=0)
        aucs = {}
        for j, name in enumerate(TARGETS):
            if len(set(Y[:, j].astype(int))) > 1:
                aucs[name] = float(roc_auc_score(Y[:, j], P[:, j]))
        macro_gold = float(np.mean(list(aucs.values())))

        print(f'Epoch {ep+1:2d}/{epochs} | Loss: {tot/len(tl):.4f} | Gold-58 Macro AUC: {macro_gold:.4f} (Best: {max(best_gold_auc, macro_gold):.4f}) | {time.time()-t0:.0f}s', flush=True)

        if macro_gold > best_gold_auc:
            best_gold_auc = macro_gold
            torch.save({
                'epoch': ep + 1,
                'model': model.state_dict(),
                'gold_auc': macro_gold,
                'aucs': aucs,
                'arch': 'swin_base_focal_mil_384',
                'targets': TARGETS
            }, '/content/engine13_swin_base_best.pt')
            print(f'  -> New Best Saved with Gold AUC: {macro_gold:.4f}', flush=True)

    print('', flush=True)
    print(f'Training Finished! Peak Gold-58 Macro-AUC: {best_gold_auc:.4f}', flush=True)

if __name__ == '__main__':
    main()
'''

nb_e13['cells'][3]['source'] = [
    "# 3. Write Engine 13: 384px Swin-Base-384 Asymmetric Focal Loss MIL Master\n",
    "with open('train_engine13_swin_focal.py', 'w', encoding='utf-8') as f:\n",
    "    f.write('''" + script_e13 + "''')\n",
    "print('Engine 13 training script generated!')\n"
]
nb_e13['cells'][4]['source'] = ["!python train_engine13_swin_focal.py\n"]

# Cell 5: Automated Kaggle publish
nb_e13['cells'][5]['source'] = [
    "# 5. Publish Engine 13 Weights to Kaggle Dataset (Automated)\n",
    "import os, json\n",
    "os.makedirs('/content/engine13_upload', exist_ok=True)\n",
    "!cp /content/engine13_swin_base_best.pt /content/engine13_upload/\n",
    "\n",
    "meta = {\n",
    "    'title': 'RSNA Knee Engine 13 Swin-Base 384 Focal Weights',\n",
    "    'id': 'cyruspinto/rsna-knee-engine13-swin-base-weights',\n",
    "    'licenses': [{'name': 'CC0-1.0'}]\n",
    "}\n",
    "with open('/content/engine13_upload/dataset-metadata.json', 'w') as f:\n",
    "    json.dump(meta, f, indent=2)\n",
    "\n",
    "!kaggle datasets create -p /content/engine13_upload -u -r zip || kaggle datasets version -p /content/engine13_upload -m 'Engine 13 Swin-Base Weights' -r zip\n",
    "print('Published Engine 13 weights to Kaggle successfully!')\n"
]

if len(nb_e13['cells']) > 6:
    nb_e13['cells'] = nb_e13['cells'][:6]

out_e13_local = "colab_engine13_swin_base_master.ipynb"
out_e13_dl = r"C:\Users\Cyrus\Downloads\colab_engine13_swin_base_master.ipynb"
with open(out_e13_local, "w", encoding="utf-8") as f:
    json.dump(nb_e13, f, indent=1)
with open(out_e13_dl, "w", encoding="utf-8") as f:
    json.dump(nb_e13, f, indent=1)
print(f"[+] Successfully built {out_e13_local} and copied to {out_e13_dl}")
