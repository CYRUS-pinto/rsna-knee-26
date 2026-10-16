"""
RSNA Knee Abnormality Detection: Engine 5 Plane-Decoupled Specialist Training (Kaggle Dual T4)
ADR-015: Eliminates Multi-Task Representation Collapse (0.9966 off-diagonal correlation).

Trains 3 Independent Plane Specialists:
1. Sagittal Specialist: Slices from Slots 0 & 1 -> [ACL, Medial Meniscus, Lateral Meniscus, Contusion]
2. Coronal Specialist: Slices from Slots 2 & 3 -> [MCL, Medial OA, Lateral OA]
3. Axial & Fluid Specialist: Slices from Slots 4 & 0 -> [PF OA, Effusion, Synovitis, Baker's, Fracture]

Evaluates Combined 12-Class Macro-AUC and Logs Prediction Correlation Matrix.
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
    'coronal': ['MCL', 'Medial OA', 'Lateral OA'],
    'axial': ['PF OA', 'Effusion', 'Synovitis', "Baker's", 'Fracture']
}

# Slot boundaries in the 96-slice Raptor volume:
# Slot 0: Sagittal T2 [0:26]
# Slot 1: Sagittal T1 [26:48]
# Slot 2: Coronal T2 [48:66]
# Slot 3: Coronal T1 [66:78]
# Slot 4: Axial [78:96]
PLANE_SLOTS = {
    'sagittal': [(0, 26, 8), (26, 48, 8)],     # 16 slices total
    'coronal': [(48, 66, 8), (66, 78, 8)],     # 16 slices total
    'axial': [(78, 96, 10), (0, 26, 6)]        # 16 slices total (Axial + fluid sensitive)
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

# ----------------- LOSS FUNCTIONS (ADR-015) -----------------
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=2.0, clip=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p_pos = probs
        p_neg = torch.clamp(1.0 - probs + self.clip, max=1.0)
        loss_pos = targets * torch.log(torch.clamp(p_pos, min=1e-7)) * ((1.0 - p_pos) ** self.gamma_pos)
        loss_neg = (1.0 - targets) * torch.log(torch.clamp(p_neg, min=1e-7)) * (probs ** self.gamma_neg)
        return - (loss_pos + loss_neg).mean()

class JointSpaceOrdinalLoss(nn.Module):
    def __init__(self, margin_weight=0.1):
        super().__init__()
        self.margin_weight = margin_weight

    def forward(self, logits, targets):
        base_loss = F.binary_cross_entropy_with_logits(logits, targets)
        if logits.size(1) >= 3:
            medial_oa, lateral_oa = logits[:, 1], logits[:, 2]
            oa_diff_target = targets[:, 1] - targets[:, 2]
            oa_diff_pred = medial_oa - lateral_oa
            hinge = F.relu(0.5 - (oa_diff_target * oa_diff_pred))
            return base_loss + self.margin_weight * hinge.mean()
        return base_loss

class CalibratedSoftBCELoss(nn.Module):
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets)

# ----------------- SPECIALIST ARCHITECTURES -----------------
class GatedAttentionPoolingHead(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=256, drop=0.1):
        super().__init__()
        self.num_classes = num_classes
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
        
    def forward(self, feats):
        B, K, D = feats.shape
        h = self.norm(feats)
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

class SpecialistMIL(nn.Module):
    def __init__(self, backbone_name='convnext_small', num_classes=4, pretrained=True, in_chans=3):
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
        self.head = GatedAttentionPoolingHead(in_features=feat_dim, num_classes=num_classes)
        
    def forward(self, x):
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        chunk_size = 16
        chunks = []
        for i in range(0, flat_x.size(0), chunk_size):
            chunks.append(self.backbone(flat_x[i:i + chunk_size]))
        feats = torch.cat(chunks, dim=0).view(B, K, -1)
        return self.head(feats)

# ----------------- DATASET -----------------
class SpecialistKaggleDataset(Dataset):
    def __init__(self, uids, labels_dict, plane, av, bv, ida, idb, res=224, is_train=True):
        self.uids = uids
        self.labels_dict = labels_dict
        self.plane = plane
        self.targets = SPECIALIST_TARGETS[plane]
        self.target_indices = [ALL_LABELS.index(t) for t in self.targets]
        self.slot_config = PLANE_SLOTS[plane]
        self.total_k = sum(n for _, _, n in self.slot_config)
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
                c = max(1, min(c, max_c)) # Strict boundary clamp
                tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], axis=0).astype(np.float32) / 255.0
                t = torch.from_numpy(tri)
                if t.shape[-1] != self.res or t.shape[-2] != self.res:
                    t = F.interpolate(t.unsqueeze(0), size=(self.res, self.res), mode='bilinear', align_corners=False)[0]
                slabs.append(t)
                
        tensor_stack = torch.stack(slabs[:self.total_k], dim=0)
        return tensor_stack, y_subset

# ----------------- TRAINING FUNCTION -----------------
def train_single_specialist(plane, train_uids, val_uids_list, labels_dict, av, bv, ida, idb,
                            epochs=6, batch_size=2, accum_steps=8, backbone_lr=1e-5, head_lr=1e-3, device='cuda'):
    targets = SPECIALIST_TARGETS[plane]
    print("\n" + "=" * 70)
    print(f"  STARTING SPECIALIST [{plane.upper()}] - {targets}")
    print(f"  Backbone LR: {backbone_lr} (Ep 0 Frozen) | Head LR: {head_lr}")
    print("=" * 70)
    
    train_ds = SpecialistKaggleDataset(train_uids, labels_dict, plane, av, bv, ida, idb, is_train=True)
    val_ds = SpecialistKaggleDataset(val_uids_list, labels_dict, plane, av, bv, ida, idb, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = SpecialistMIL(backbone_name='convnext_small', num_classes=len(targets), pretrained=True).to(device)
    
    if plane == 'sagittal':
        loss_fn = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=2.0)
    elif plane == 'coronal':
        loss_fn = JointSpaceOrdinalLoss(margin_weight=0.1)
    else:
        loss_fn = CalibratedSoftBCELoss()
        
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': backbone_lr},
        {'params': model.head.parameters(), 'lr': head_lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    
    best_macro_auc = 0.0
    best_preds = None
    
    for epoch in range(epochs):
        t0 = time.time()
        if epoch == 0:
            for p in model.backbone.parameters():
                p.requires_grad = False
            print("  [Warmup Ep 0]: Backbone frozen. Optimizing attention head only.")
        elif epoch == 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print("  [Ep 1+ Full]: Backbone unfrozen at differential LR.")
            
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                    loss = loss_fn(logits, y) / accum_steps
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model(x)
                loss = loss_fn(logits, y) / accum_steps
                loss.backward()
                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            train_loss += loss.item() * accum_steps
            
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation Loop
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model(x)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(y.numpy())
                
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        class_aucs = {}
        for j, name in enumerate(targets):
            y_t = (val_targets[:, j] >= 0.5).astype(int)
            class_aucs[name] = exact_roc_auc(y_t, val_preds[:, j])
            
        macro_auc = float(np.mean(list(class_aucs.values())))
        ep_time = time.time() - t0
        print(f"Epoch {epoch+1:02d}/{epochs:02d} [{ep_time:.1f}s] - Loss: {train_loss:.4f} | Val AUC: {macro_auc:.5f}")
        for name, a in class_aucs.items():
            print(f"   - {name:20s}: {a:.4f}")
            
        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            best_preds = val_preds
            out_dir = '/kaggle/working/weights' if os.path.exists('/kaggle/working') else 'weights'
            os.makedirs(out_dir, exist_ok=True)
            weight_path = os.path.join(out_dir, f"engine5_{plane}_fold0_best.pth")
            torch.save(model.state_dict(), weight_path)
            print(f">> Saved Best [{plane.upper()}] Weights to {weight_path} (AUC: {macro_auc:.5f}) <<")
            
    return best_macro_auc, best_preds

# ----------------- MAIN PIPELINE -----------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== Starting Engine 5 Plane-Decoupled Training on {device} ===")
    
    av = _find('all_vols.npy')
    bv = _find('extra_vols.npy')
    ida = _find('all_ids.npy')
    idb = _find('extra_ids.npy')
    lbl_file = _find('SUPER_CONSENSUS_labels.parquet', 'SUPER_CONSENSUS_labels.csv')
    
    print(f"Dataset files: av={av}, bv={bv}, ida={ida}, lbl={lbl_file}")
    
    df_labels = pd.read_parquet(lbl_file) if lbl_file.endswith('.parquet') else pd.read_csv(lbl_file)
    labels_dict = {row['StudyInstanceUID']: row[ALL_LABELS].values.astype(np.float32) for _, row in df_labels.iterrows()}
    
    # Stratified Split on Fold 0
    np.random.seed(42)
    all_uids = df_labels['StudyInstanceUID'].tolist()
    np.random.shuffle(all_uids)
    val_count = len(all_uids) // 5
    val_uids = set(all_uids[:val_count])
    train_uids = [u for u in all_uids if u not in val_uids]
    val_uids_list = sorted(list(val_uids))
    
    print(f"Cohorts: Train={len(train_uids)} | Val={len(val_uids_list)}")
    
    all_oof = {}
    for plane in ['sagittal', 'coronal', 'axial']:
        auc, preds = train_single_specialist(
            plane=plane,
            train_uids=train_uids,
            val_uids_list=val_uids_list,
            labels_dict=labels_dict,
            av=av, bv=bv, ida=ida, idb=idb,
            epochs=6, batch_size=2, accum_steps=8,
            device=device
        )
        for j, col in enumerate(SPECIALIST_TARGETS[plane]):
            all_oof[col] = preds[:, j]
            
    # Combine & Evaluate 12 Classes
    df_oof = pd.DataFrame(all_oof)
    df_oof['StudyInstanceUID'] = val_uids_list
    oof_dir = '/kaggle/working/oof' if os.path.exists('/kaggle/working') else 'oof'
    os.makedirs(oof_dir, exist_ok=True)
    combined_path = os.path.join(oof_dir, 'oof_engine5_combined_fold0.parquet')
    df_oof.to_parquet(combined_path, index=False)
    print(f">> Saved Combined 12-Class OOF to {combined_path} <<")
    
    print("\n" + "=" * 70)
    print("  ENGINE 5 OVERALL 12-CLASS PERFORMANCE")
    print("=" * 70)
    val_targets_df = df_labels[df_labels['StudyInstanceUID'].isin(val_uids_list)].sort_values('StudyInstanceUID')
    df_oof_sorted = df_oof.sort_values('StudyInstanceUID')
    
    aucs = {}
    for col in ALL_LABELS:
        y_t = (val_targets_df[col].values >= 0.5).astype(int)
        y_p = df_oof_sorted[col].values
        a = exact_roc_auc(y_t, y_p)
        aucs[col] = a
        print(f"  {col:22s}: {a:.4f}")
        
    macro_12 = float(np.mean(list(aucs.values())))
    print("---------------------------------------")
    print(f"  ENGINE 5 FULL MACRO-AUC: {macro_12:.5f}")
    print("---------------------------------------")
    
    corr_matrix = df_oof_sorted[ALL_LABELS].corr().values
    np.fill_diagonal(corr_matrix, 0.0)
    mean_corr = float(np.mean(np.abs(corr_matrix)))
    print(f"  MEAN OFF-DIAGONAL CORRELATION: {mean_corr:.4f} (Anti-Collapse Target: <= 0.40)")
    print("=" * 70)

if __name__ == '__main__':
    main()
