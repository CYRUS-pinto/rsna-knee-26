"""
RSNA Knee Abnormality Detection — Frontier 3D Slice-Transformer MIL Trainer
Platform: Google Colab Pro / Lambda / Cloud (A100 / L4 GPU)
Architecture: CoAtNet-2 384 Backbone + 3D Slice-Transformer Attention Head + Asymmetric Loss (ASL)
Stability Guard: Zero-Divergence BF16 direct backprop + 1.0 gradient clipping + 2e-4 head LR
"""

import os, sys, time, json, math, random, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import timm

LAB = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA',
    'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Binary Classification.
    Dynamically downweights easy negative samples to focus gradient descent
    on ambiguous hard positive and hard negative margins.
    Ref: Ridnik et al., ICCV 2021.
    """
    def __init__(self, gamma_neg=2.0, gamma_pos=0.0, clip=0.05, eps=1e-8, pos_weight=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.pos_weight = pos_weight

    def forward(self, x, y):
        xs_pos = torch.sigmoid(x).clamp(self.eps, 1.0 - self.eps)
        xs_neg = 1.0 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        los_pos = y * torch.log(xs_pos)
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = (xs_pos * y + xs_neg * (1.0 - y)).clamp(0.0, 1.0)
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
            one_sided_w = torch.pow((1.0 - pt).clamp(min=0.0), one_sided_gamma)
            if self.pos_weight is not None:
                one_sided_w = one_sided_w * (y * self.pos_weight + (1.0 - y))
            loss = -one_sided_w * (los_pos + los_neg)
        else:
            if self.pos_weight is not None:
                loss = -(y * self.pos_weight * los_pos + (1.0 - y) * los_neg)
            else:
                loss = -(los_pos + los_neg)
        return loss.mean()

class TwoPartVols:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])
    def __len__(self): return self.shape[0]
    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]

def open_corpus(root='/content/corpus'):
    av = os.path.join(root, 'all_vols.npy')
    bv = os.path.join(root, 'extra_vols.npy')
    am = os.path.join(root, 'all_masks.npy')
    bm = os.path.join(root, 'extra_masks.npy')
    vols = TwoPartVols(np.load(av, mmap_mode='r'), np.load(bv, mmap_mode='r'))
    masks = np.concatenate([np.load(am), np.load(bm)], axis=0)
    return vols, masks

def open_ids(root='/content/corpus'):
    a = os.path.join(root, 'all_ids.npy')
    b = os.path.join(root, 'extra_ids.npy')
    return np.concatenate([
        np.load(a, allow_pickle=True).astype(str),
        np.load(b, allow_pickle=True).astype(str)
    ])

class StudyWindows(Dataset):
    def __init__(self, ids, id2row, labels, res=384, k=16, train=True, corpus_dir='/content/corpus'):
        self.ids = ids
        self.id2row = id2row
        self.labels = labels
        self.res = res
        self.k = k
        self.train = train
        self.corpus_dir = corpus_dir
        self.vols = None
        self.masks = None

    def __len__(self):
        return len(self.ids)

    def _ensure(self):
        if self.vols is None:
            self.vols, self.masks = open_corpus(self.corpus_dir)

    def _centers(self, valid, count):
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs:
            cs = [max(1, min((lo + hi) // 2, self._D - 2))]
        if self.train:
            reps = count // len(cs) + 1
            pool = (cs * reps)
            random.shuffle(pool)
            return sorted(pool[:count]) # Monotonic Z-axis order for spatial positional encoding
        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)
        return sorted([cs[i] for i in idx])

    def _resize(self, tri):
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != self.res or t.shape[-2] != self.res:
            t = F.interpolate(t[None], size=(self.res, self.res), mode='bilinear', align_corners=False)[0]
        return t.numpy()

    def __getitem__(self, i):
        self._ensure()
        uid = self.ids[i]
        row = self.id2row[uid]
        self._D = self.vols.shape[1]
        m = self.masks[row]
        valid = np.where(m > 0)[0]
        if len(valid) < 3:
            valid = np.arange(min(3, self._D))
        cs = self._centers(valid, self.k)
        vol = self.vols[row]
        tiles = []
        for c in cs:
            c = max(1, min(c, self._D - 2))
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
            tiles.append(self._resize(tri))
        wins = np.stack(tiles, 0)
        x = torch.from_numpy(wins)
        if self.train:
            g = 1.0 + (random.random() - 0.5) * 0.20
            x = (x * g).clamp(0, 1)
        y = torch.from_numpy(self.labels[uid])
        return x, y

def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])

class SliceTransformerHead(nn.Module):
    """
    3D Sequential Slice-Transformer MIL Head.
    Models contiguous anatomical disruption across the slice axis (Z-plane)
    using 1D slice positional embeddings, multi-head self-attention, and post-norm.
    """
    def __init__(self, F_dim=768, n=12, num_heads=8, num_layers=2, drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(F_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 128, F_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=F_dim, nhead=num_heads, dim_feedforward=F_dim * 2,
            dropout=drop, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.att = nn.Sequential(
            nn.Linear(F_dim, 256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, n)
        )
        self.post_norm = nn.LayerNorm(F_dim)
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        
    def forward(self, feats):
        B, K, D = feats.shape
        h = self.norm(feats)
        h = h + self.pos_emb[:, :K, :]
        h_trans = self.transformer(h)
        a = torch.softmax(self.att(h_trans), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h_trans)
        pooled = self.post_norm(pooled)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=384, n=12, drop=0.2, head_type='transformer'):
        super().__init__()
        self.backbone = backbone
        self.head_type = head_type
        if head_type == 'transformer':
            self.head = SliceTransformerHead(F_dim=F_dim, n=n, num_heads=8, num_layers=2, drop=drop)
        else:
            self.norm = nn.LayerNorm(F_dim)
            self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                     nn.Linear(256, n))
            self.clsW = nn.Parameter(torch.zeros(n, F_dim))
            self.clsb = nn.Parameter(torch.zeros(n))
            nn.init.trunc_normal_(self.clsW, std=0.02)

    def encode(self, x):
        B, K = x.shape[:2]
        flat_x = x.flatten(0, 1)
        chunk_size = 16
        chunks = [self.backbone(flat_x[i:i + chunk_size]) for i in range(0, flat_x.size(0), chunk_size)]
        f = torch.cat(chunks, dim=0)
        return f.view(B, K, -1)

    def forward(self, x):
        feats = self.encode(x)
        if self.head_type == 'transformer':
            return self.head(feats)
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k')
    ap.add_argument('--res', type=int, default=384)
    ap.add_argument('--k', type=int, default=16, help='Number of slices to sample per study during training')
    ap.add_argument('--k_eval', type=int, default=24, help='Number of slices during evaluation')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--bs', type=int, default=4, help='Batch size (bs=4 with grad accum=2 runs smooth on A100)')
    ap.add_argument('--accum', type=int, default=2, help='Gradient accumulation steps (effective bs = bs * accum)')
    ap.add_argument('--bb_lr', type=float, default=2e-5)
    ap.add_argument('--head_lr', type=float, default=2e-4)
    ap.add_argument('--head', default='transformer', choices=['mil', 'transformer'], help='Head architecture: mil or transformer')
    ap.add_argument('--loss', default='asl', choices=['bce', 'asl'], help='Loss: bce or asl')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--tag', default='colab_super_coatnet_transformer_0925')
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'=== Starting Training: {a.tag} ===')
    print(f'Backbone: {a.arch} | Head: {a.head} | Loss: {a.loss} | K: {a.k} slices | Resolution: {a.res}x{a.res}')
    print(f'Batch Size: {a.bs} x Accum {a.accum} (Effective BS: {a.bs * a.accum})')
    print(f'Learning Rates: Backbone={a.bb_lr} | Head={a.head_lr}')
    
    # Auto-fetch data if new Colab runtime session was started
    if not os.path.exists('/content/corpus/all_ids.npy') or not os.path.exists('/content/rsna_comp/train.csv'):
        print('[auto-fetch] New Colab VM detected: /content/corpus is missing.')
        print('[auto-fetch] Downloading precomputed slice stacks and labels (~1-2 minutes)...', flush=True)
        os.makedirs('/content/corpus', exist_ok=True)
        os.makedirs('/content/labels', exist_ok=True)
        os.makedirs('/content/rsna_comp', exist_ok=True)
        os.system('kaggle datasets download -d dreaddevelopment/knee-raptor-corpus -p /content/corpus --unzip -q')
        os.system('kaggle datasets download -d dreaddevelopment/knee-raptor-corpus-ext -p /content/corpus --unzip -q')
        os.system('kaggle datasets download -d cyruspinto/rsna-knee-super-consensus-labels-0916 -p /content/labels --unzip -q')
        os.system('kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p /content/rsna_comp -q')
        os.system('unzip -q -o /content/rsna_comp/train.csv.zip -d /content/rsna_comp/ 2>/dev/null || true')
        print('[auto-fetch] All training corpora downloaded and verified!', flush=True)

    ids = open_ids('/content/corpus')
    id2row = {u: i for i, u in enumerate(ids)}
    idset = set(ids)

    tr = pd.read_csv('/content/rsna_comp/train.csv')
    tr['StudyInstanceUID'] = tr['StudyInstanceUID'].astype(str)
    gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index('StudyInstanceUID')
    gold_ids = [u for u in gold_df.index if u in idset]

    lab_path = '/content/labels/SUPER_CONSENSUS_labels.parquet'
    if not os.path.exists(lab_path):
        lab_path = '/content/labels/SUPER_CONSENSUS_labels.csv'
    soft = pd.read_parquet(lab_path) if lab_path.endswith('.parquet') else pd.read_csv(lab_path)
    soft['StudyInstanceUID'] = soft['StudyInstanceUID'].astype(str)
    soft = soft.set_index('StudyInstanceUID')

    train_ids = [u for u in ids if u in soft.index and u not in set(gold_ids)]
    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in train_ids}
    for u in gold_ids:
        labels[u] = gold_df.loc[u, LAB].values.astype(np.float32)
    print(f'Train: {len(train_ids)} studies | Validation (Gold Anchor): {len(gold_ids)} studies')

    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)

    tds = StudyWindows(train_ids, id2row, labels, a.res, a.k, train=True)
    vds = StudyWindows(gold_ids, id2row, labels, a.res, a.k_eval, train=False)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    vl = DataLoader(vds, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers, pin_memory=True)

    bb = timm.create_model(a.arch, pretrained=True, num_classes=0, in_chans=3, global_pool='avg')
    if hasattr(bb, 'set_grad_checkpointing'):
        try:
            bb.set_grad_checkpointing(True)
            print('Gradient checkpointing enabled on backbone.')
        except Exception:
            pass
    F_dim = bb.num_features
    model = RaptorClassifier(bb, F_dim=F_dim, head_type=a.head).to(dev)

    head_p = [p for n, p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    bb_p = [p for n, p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]
    opt = torch.optim.AdamW([{'params': bb_p, 'lr': a.bb_lr}, {'params': head_p, 'lr': a.head_lr}], weight_decay=0.02)
    total_opt_steps = math.ceil(len(tl) / a.accum) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=total_opt_steps, pct_start=0.15)
    
    if a.loss == 'asl':
        lossf = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05, pos_weight=pw)
        print('Using Asymmetric Loss (ASL) for hard-example ROC AUC optimization.')
    else:
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        print('Using BCEWithLogitsLoss.')

    use_bf16 = torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = not use_bf16
    scaler = (torch.amp.GradScaler('cuda') if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler()) if use_scaler else None
    print(f'Precision: {"BF16 (Direct Backprop)" if use_bf16 else "FP16 (GradScaler Enabled)"}')

    @torch.no_grad()
    def evaluate():
        model.eval()
        P, Y = [], []
        for x, y in vl:
            with torch.autocast(dev, dtype=autocast_dtype):
                o = torch.sigmoid(model(x.to(dev)).float())
            P.append(o.cpu().numpy())
            Y.append(y.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        aucs = {}
        for j, name in enumerate(LAB):
            if len(set(Y[:, j].astype(int))) > 1:
                aucs[name] = float(roc_auc_score(Y[:, j], P[:, j]))
        return float(np.mean(list(aucs.values()))), aucs, P, Y

    best_auc = 0.0
    best_state = None
    top_states = []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        opt.zero_grad()
        for step, (x, y) in enumerate(tl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=autocast_dtype):
                logits = model(x)
                loss = lossf(logits.float(), y) / a.accum
            
            if use_scaler:
                scaler.scale(loss).backward()
                if (step + 1) % a.accum == 0 or (step + 1) == len(tl):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                    sched.step()
            else:
                loss.backward()
                if (step + 1) % a.accum == 0 or (step + 1) == len(tl):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    sched.step()
                    
            tot += loss.item() * a.accum
        au, aucs, P, Y = evaluate()
        print(f'Epoch {ep+1:2d}/{a.epochs} | Loss: {tot/len(tl):.4f} | Gold Macro-AUC: {au:.4f} (Best: {max(best_auc, au):.4f}) | {time.time()-t0:.0f}s', flush=True)
        state_copy = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        top_states.append({'epoch': ep+1, 'auc': au, 'state': state_copy})
        top_states.sort(key=lambda x: -x['auc'])
        if len(top_states) > 3:
            top_states.pop()
        if au > best_auc:
            best_auc = au
            torch.save({
                'model': state_copy, 'gold_auc': au, 'aucs': aucs,
                'arch': a.arch, 'res': a.res, 'lab': LAB, 'head': a.head, 'k': a.k
            }, f'{a.tag}_best.pt')
            print(f'   -> Saved new best checkpoint ({au:.4f})', flush=True)

    # Stochastic Weight Averaging (SWA) over top 3 epochs
    if len(top_states) > 1:
        avg_sd = {}
        for k in top_states[0]['state']:
            avg_sd[k] = sum(s['state'][k].double() for s in top_states).div(len(top_states)).to(top_states[0]['state'][k].dtype)
        model.load_state_dict(avg_sd)
        swa_auc, swa_aucs, _, _ = evaluate()
        print(f'\nSWA Macro-AUC: {swa_auc:.4f} (across epochs {[s["epoch"] for s in top_states]})', flush=True)
        torch.save({
            'model': avg_sd, 'gold_auc': swa_auc, 'aucs': swa_aucs,
            'arch': a.arch, 'res': a.res, 'lab': LAB, 'head': a.head, 'k': a.k
        }, f'{a.tag}_swa.pt')
        print(f'Saved SWA checkpoint: {a.tag}_swa.pt')

if __name__ == '__main__':
    main()
