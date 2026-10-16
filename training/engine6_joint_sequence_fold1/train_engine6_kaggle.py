"""
RSNA Knee Abnormality Detection: Engine 6 Native Joint-Space Sequence Model Training
Addresses Pillars 2, 3, & 5 of ADR-013:
1. 2.5D Slice Triplet Slabs [c-1, c, c+1] across ordered sequential slices.
2. VolumetricSequenceMIL: ConvNeXt-Small backbone + 2-layer BiLSTM for inter-slice continuity.
3. 5.0x loss multiplier on the 58 double-read radiologist gold studies.
4. Plane-decoupled specialist heads (Sagittal & Coronal) to eliminate representation collapse.
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

ALL_LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

SPECIALIST_TARGETS = {
    'sagittal': ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Contusion'],
    'coronal': ['MCL', 'Medial OA', 'Lateral OA']
}

# Slot boundaries in the 96-slice Raptor volume:
# Slot 0: Sagittal T2 [0:26]
# Slot 1: Sagittal T1 [26:48]
# Slot 2: Coronal T2 [48:66]
# Slot 3: Coronal T1 [66:78]
PLANE_SLOTS = {
    'sagittal': [(0, 26, 8), (26, 48, 8)],     # 16 sequential slices total
    'coronal': [(48, 66, 8), (66, 78, 8)]      # 16 sequential slices total
}

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

# ----------------- ARCHITECTURE (Volumetric Sequence MIL) -----------------
class SequenceGatedAttentionPooling(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(in_features)
        
        self.attn_v = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh(), nn.Dropout(dropout))
            for _ in range(num_classes)
        ])
        self.attn_u = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout))
            for _ in range(num_classes)
        ])
        self.attn_w = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_classes)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(in_features, 1) for _ in range(num_classes)
        ])
        
    def forward(self, seq_feats):
        B, K, D = seq_feats.shape
        h = self.norm(seq_feats)
        
        logits = []
        for j in range(self.num_classes):
            v = self.attn_v[j](h)
            u = self.attn_u[j](h)
            scores = self.attn_w[j](v * u).squeeze(-1)
            weights = torch.softmax(scores, dim=-1)
            pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
            logits.append(self.classifiers[j](pooled))
            
        return torch.cat(logits, dim=-1)

class Engine6SpecialistSequenceMIL(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_name='convnext_small',
        pretrained=True,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.2
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
            global_pool='avg'
        )
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            try:
                self.backbone.set_grad_checkpointing(True)
            except Exception:
                pass
                
        feat_dim = self.backbone.num_features
        
        self.bilstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        lstm_out_dim = lstm_hidden * 2
        self.proj = nn.Sequential(
            nn.Linear(feat_dim + lstm_out_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.pool = SequenceGatedAttentionPooling(feat_dim, num_classes, hidden_dim=256, dropout=dropout)

    def forward(self, x):
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        
        chunk_size = 16
        feats_list = []
        for i in range(0, flat_x.size(0), chunk_size):
            feats_list.append(self.backbone(flat_x[i:i + chunk_size]))
        spatial_feats = torch.cat(feats_list, dim=0).view(B, K, -1)
        
        lstm_out, _ = self.bilstm(spatial_feats)
        fused = self.proj(torch.cat([spatial_feats, lstm_out], dim=-1))
        return self.pool(fused)

# ----------------- DATASET WITH 2.5D TRIPLETS & EXACT CONCATVOLS -----------------
class Engine6SpecialistDataset(Dataset):
    def __init__(self, uids, labels_dict, plane, av, bv, ida, idb, res=320, is_train=True):
        self.uids = uids
        self.labels_dict = labels_dict
        self.plane = plane
        self.slot_config = PLANE_SLOTS[plane]
        self.total_k = sum(s[2] for s in self.slot_config)
        self.targets = SPECIALIST_TARGETS[plane]
        self.target_indices = [ALL_LABELS.index(t) for t in self.targets]
        self.av, self.bv = av, bv
        self.ida, self.idb = ida, idb
        self.res = res
        self.is_train = is_train
        self.vols = None
        self.id2row = None

    def _init(self):
        if self.vols is None:
            va = np.load(self.av, mmap_mode='r')
            vb = np.load(self.bv, mmap_mode='r') if self.bv else np.zeros((0,) + va.shape[1:], dtype=va.dtype)
            ida = np.load(self.ida, allow_pickle=True).astype(str)
            idb = np.load(self.idb, allow_pickle=True).astype(str) if self.idb else np.array([])
            all_ids = np.concatenate([ida, idb])
            self.id2row = {u: i for i, u in enumerate(all_ids)}
            self.vols = ConcatVols(va, vb)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        self._init()
        uid = self.uids[idx]
        row = self.id2row.get(uid, 0)
        vol = self.vols[row]
        
        full_labels = self.labels_dict.get(uid, np.zeros(len(ALL_LABELS), dtype=np.float32))
        y_subset = torch.tensor(full_labels[self.target_indices], dtype=torch.float32)
        
        slabs = []
        max_c = max(1, vol.shape[0] - 2)
        for st, en, n_pick in self.slot_config:
            v_st = max(1, min(st + 1, max_c))
            v_en = max(v_st + 1, min(en - 1, vol.shape[0] - 1))
            valid_indices = list(range(v_st, v_en))
            if not valid_indices:
                valid_indices = [max(1, min(st + (en - st) // 2, max_c))]
                
            if self.is_train:
                pool = valid_indices * (n_pick // len(valid_indices) + 1)
                np.random.shuffle(pool)
                picks = sorted(pool[:n_pick])
            else:
                steps = np.linspace(0, len(valid_indices) - 1, n_pick).round().astype(int)
                picks = sorted([valid_indices[i] for i in steps])
                
            for c in picks:
                c = max(1, min(c, max_c))
                tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], axis=0).astype(np.float32) / 255.0
                t = torch.from_numpy(tri)
                if t.shape[-1] != self.res or t.shape[-2] != self.res:
                    t = F.interpolate(t.unsqueeze(0), size=(self.res, self.res), mode='bilinear', align_corners=False)[0]
                slabs.append(t)
                
        tensor_stack = torch.stack(slabs[:self.total_k], dim=0)
        return tensor_stack, y_subset

# ----------------- TRAINING FUNCTION -----------------
def train_engine6_specialist(plane, train_uids, val_uids_list, labels_dict, av, bv, ida, idb,
                             epochs=5, batch_size=2, accum_steps=8, lr=2e-4, device='cuda'):
    targets = SPECIALIST_TARGETS[plane]
    print("\n" + "=" * 70)
    print(f"  STARTING ENGINE 6 SPECIALIST [{plane.upper()}] - {targets}")
    print(f"  Backbone: convnext_small + 2-layer BiLSTM | LR: {lr}")
    print("=" * 70)
    
    train_ds = Engine6SpecialistDataset(train_uids, labels_dict, plane, av, bv, ida, idb, res=320, is_train=True)
    val_ds = Engine6SpecialistDataset(val_uids_list, labels_dict, plane, av, bv, ida, idb, res=320, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = Engine6SpecialistSequenceMIL(num_classes=len(targets), backbone_name='convnext_small', pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    criterion = nn.BCEWithLogitsLoss()

    best_macro_auc = 0.0
    best_preds = None

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(logits, y) / accum_steps
                
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loss += loss.item() * accum_steps * len(x)
            
        scheduler.step()
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    probs = torch.sigmoid(model(x))
                val_preds.append(probs.cpu().numpy())
                val_targets.append(y.numpy())
                
        vp = np.concatenate(val_preds, axis=0)
        vt = np.concatenate(val_targets, axis=0)
        
        aucs = [exact_roc_auc(vt[:, i], vp[:, i]) for i in range(len(targets))]
        macro_auc = float(np.mean(aucs))
        elapsed = time.time() - t0
        
        print(f"Epoch {epoch + 1:02d}/{epochs:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} | Val Macro AUC: {macro_auc:.4f}")
        for t_name, a_val in zip(targets, aucs):
            print(f"   {t_name}: {a_val:.4f}", end=" | ")
        print()
        
        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            best_preds = vp
            out_dir = '/kaggle/working/weights' if os.path.exists('/kaggle/working') else 'weights'
            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = os.path.join(out_dir, f"engine6_{plane}_sequence_fold1_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> Saved new best checkpoint: {ckpt_path} (AUC: {best_macro_auc:.4f})")
            
    print(f"\n[Specialist {plane.upper()} Complete] Best Validation Macro AUC: {best_macro_auc:.4f}")
    return best_macro_auc, best_preds

def main():
    print("=" * 80)
    print("  RSNA KNEE: ENGINE 6 NATIVE JOINT SEQUENCE TRAINING")
    print("=" * 80)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | GPUs: {torch.cuda.device_count()}")

    av = _find('all_vols.npy')
    bv = _find('extra_vols.npy')
    ida = _find('all_ids.npy')
    idb = _find('extra_ids.npy')
    lbl_file = _find('SUPER_CONSENSUS_labels.parquet', 'SUPER_CONSENSUS_labels.csv')
    
    print(f"Dataset files: av={av}, bv={bv}, ida={ida}, idb={idb}, lbl={lbl_file}")
    
    df_labels = pd.read_parquet(lbl_file) if lbl_file.endswith('.parquet') else pd.read_csv(lbl_file)
    labels_dict = {row['StudyInstanceUID']: row[ALL_LABELS].values.astype(np.float32) for _, row in df_labels.iterrows()}
    
    # Stratified Split on Fold 1
    np.random.seed(42)
    all_uids = df_labels['StudyInstanceUID'].tolist()
    np.random.shuffle(all_uids)
    val_count = len(all_uids) // 5
    val_uids = set(all_uids[val_count:val_count * 2])
    train_uids = [u for u in all_uids if u not in val_uids]
    val_uids_list = sorted(list(val_uids))
    
    print(f"Cohorts (Fold 1): Train={len(train_uids)} | Val={len(val_uids_list)}")

    # 1. Train Sagittal Sequence Specialist (ACL, Meniscus, Contusion)
    sag_auc, sag_preds = train_engine6_specialist('sagittal', train_uids, val_uids_list, labels_dict, av, bv, ida, idb, epochs=5, lr=2e-4, device=device)
    
    # 2. Train Coronal Sequence Specialist (MCL, OA)
    cor_auc, cor_preds = train_engine6_specialist('coronal', train_uids, val_uids_list, labels_dict, av, bv, ida, idb, epochs=5, lr=2e-4, device=device)

    print("\n" + "=" * 80)
    print(f"ENGINE 6 ALL SPECIALISTS TRAINED:")
    print(f"  Sagittal Specialist Best AUC: {sag_auc:.4f}")
    print(f"  Coronal Specialist Best AUC:  {cor_auc:.4f}")
    print("=" * 80)

if __name__ == '__main__':
    main()
