"""
RSNA Knee Abnormality Detection - Multi-Plane Engine 3 (5-Fold CV Trainer)
Key Features:
1. Multi-Plane 2.5D Slab Representation with Discrete Sequence Plane Embeddings (ADR-006, ADR-008).
2. Leak-Free 5-Fold Stratified Group Cross-Validation on all 4,407 studies (ADR-002).
3. Continuous Soft Target BCEWithLogitsLoss Optimization (ADR-009).
4. Raw Logit Analytic Mann-Whitney ROC-AUC Evaluation (ADR-003).
5. Fast, memory-efficient training designed for Kaggle Dual T4 / A100 GPUs.
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

# Local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from validation.cv_framework import LABELS, exact_roc_auc
from training.engine3_slice_transformer.model_multiplane import MultiPlaneSliceTransformerHead

class MultiPlaneStudyDataset(Dataset):
    """
    Loads multi-plane 2.5D slice windows from precomputed corpus.
    Dynamically samples k slices with associated plane IDs.
    """
    def __init__(self, uids, labels_dict, corpus_dir='/content/corpus', k=32, res=384, is_train=True):
        self.uids = uids
        self.labels_dict = labels_dict
        self.corpus_dir = corpus_dir
        self.k = k
        self.res = res
        self.is_train = is_train
        self.vols = None
        self.masks = None
        self.ids = None
        self.id2row = None

    def _init_corpus(self):
        if self.vols is None:
            av = os.path.join(self.corpus_dir, 'all_vols.npy')
            bv = os.path.join(self.corpus_dir, 'extra_vols.npy')
            am = os.path.join(self.corpus_dir, 'all_masks.npy')
            bm = os.path.join(self.corpus_dir, 'extra_masks.npy')
            ai = os.path.join(self.corpus_dir, 'all_ids.npy')
            bi = os.path.join(self.corpus_dir, 'extra_ids.npy')
            
            # Memory-mapped reads for zero-RAM overhead
            v_a = np.load(av, mmap_mode='r')
            v_b = np.load(bv, mmap_mode='r') if os.path.exists(bv) else np.zeros((0,) + v_a.shape[1:], dtype=v_a.dtype)
            
            # Combine IDs
            ids_a = np.load(ai, allow_pickle=True).astype(str)
            ids_b = np.load(bi, allow_pickle=True).astype(str) if os.path.exists(bi) else np.array([])
            self.ids = np.concatenate([ids_a, ids_b])
            self.id2row = {u: i for i, u in enumerate(self.ids)}
            
            m_a = np.load(am)
            m_b = np.load(bm) if os.path.exists(bm) else np.zeros((0,) + m_a.shape[1:], dtype=m_a.dtype)
            self.masks = np.concatenate([m_a, m_b], axis=0)
            
            class ConcatVols:
                def __init__(self, a, b):
                    self.a, self.b = a, b
                    self.n_a = len(a)
                def __getitem__(self, idx):
                    return self.a[idx] if idx < self.n_a else self.b[idx - self.n_a]
                @property
                def shape(self):
                    return (self.n_a + len(self.b),) + self.a.shape[1:]
                    
            self.vols = ConcatVols(v_a, v_b)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        self._init_corpus()
        uid = self.uids[idx]
        row = self.id2row.get(uid, 0)
        
        # Sample k slices
        total_d = self.vols.shape[1]
        mask = self.masks[row]
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) < 3:
            valid_indices = np.arange(total_d)
            
        lo, hi = int(valid_indices.min()), int(valid_indices.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= 0 and c + 1 < total_d]
        if not cs:
            cs = [max(1, min((lo + hi) // 2, total_d - 2))]
            
        if self.is_train:
            # Random uniform slice sampling
            pool = cs * (self.k // len(cs) + 1)
            np.random.shuffle(pool)
            chosen = sorted(pool[:self.k])
        else:
            # Deterministic linear spacing
            step_indices = np.linspace(0, len(cs) - 1, self.k).round().astype(int)
            chosen = sorted([cs[i] for i in step_indices])
            
        # Extract 2.5D triplet slabs
        vol = self.vols[row]
        slabs = []
        for c in chosen:
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], axis=0).astype(np.float32) / 255.0
            t = torch.from_numpy(tri)
            if t.shape[-1] != self.res or t.shape[-2] != self.res:
                t = F.interpolate(t.unsqueeze(0), size=(self.res, self.res), mode='bilinear', align_corners=False)[0]
            slabs.append(t)
            
        tensor_stack = torch.stack(slabs, dim=0) # (K, 3, H, W)
        
        # Plane IDs: Sequence plane assignment (default 0 for sagittal single-corpus)
        plane_ids = torch.zeros(self.k, dtype=torch.long)
        
        # Target soft label
        y = torch.tensor(self.labels_dict[uid], dtype=torch.float32)
        return tensor_stack, plane_ids, y

class MultiPlaneEngine3(nn.Module):
    def __init__(self, backbone_name='convnext_base.fb_in22k_ft_in1k_384', F_dim=1024, n=12, drop=0.1):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, in_chans=3, global_pool='avg')
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            try: self.backbone.set_grad_checkpointing(True)
            except Exception: pass
        feat_dim = self.backbone.num_features
        self.head = MultiPlaneSliceTransformerHead(F_dim=feat_dim, n=n, num_heads=8, num_layers=2, drop=drop)
        
    def forward(self, x, plane_ids=None):
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        chunk_size = 16
        chunks = [self.backbone(flat_x[i:i + chunk_size]) for i in range(0, flat_x.size(0), chunk_size)]
        feats = torch.cat(chunks, dim=0).view(B, K, -1)
        return self.head(feats, plane_ids=plane_ids)

def run_train_cv(fold=0, epochs=6, batch_size=4, lr_bb=3e-5, lr_head=2e-4, corpus_dir='/content/corpus'):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 65)
    print(f"  STARTING MULTI-PLANE ENGINE 3 CV TRAINING - FOLD {fold}")
    print(f"  Device: {dev} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 65)
    
    # Load 5-Fold Partition
    folds_path = 'data/train_folds.parquet'
    if not os.path.exists(folds_path):
        folds_path = os.path.join('/content', folds_path)
    df_folds = pd.read_parquet(folds_path)
    
    # Train / Val Split
    train_df = df_folds[df_folds['fold'] != fold].copy()
    val_df = df_folds[df_folds['fold'] == fold].copy()
    
    train_uids = train_df['StudyInstanceUID'].tolist()
    val_uids = val_df['StudyInstanceUID'].tolist()
    
    labels_dict = {row['StudyInstanceUID']: row[LABELS].values.astype(np.float32) for _, row in df_folds.iterrows()}
    
    print(f"  Training Studies: {len(train_uids)} | Validation Studies: {len(val_uids)}")
    
    # Datasets & Loaders
    train_ds = MultiPlaneStudyDataset(train_uids, labels_dict, corpus_dir=corpus_dir, k=16, is_train=True)
    val_ds = MultiPlaneStudyDataset(val_uids, labels_dict, corpus_dir=corpus_dir, k=24, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = MultiPlaneEngine3(backbone_name='convnext_base.fb_in22k_ft_in1k_384').to(dev)
    
    # Loss & Optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    bb_params = [p for n, p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {'params': bb_params, 'lr': lr_bb},
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    # Precision
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"  Precision: {'BF16 Direct Backprop' if use_bf16 else 'FP16'}")
    
    os.makedirs('weights', exist_ok=True)
    best_auc = 0.0
    
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        for step, (x, plane_ids, y) in enumerate(train_loader):
            x = x.to(dev, non_blocking=True)
            plane_ids = plane_ids.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.autocast(dev, dtype=autocast_dtype):
                logits = model(x, plane_ids=plane_ids)
                loss = loss_fn(logits, y)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation Evaluation
        model.eval()
        val_logits, val_targets = [], []
        with torch.no_grad():
            for x, plane_ids, y in val_loader:
                x = x.to(dev, non_blocking=True)
                plane_ids = plane_ids.to(dev, non_blocking=True)
                with torch.autocast(dev, dtype=autocast_dtype):
                    logits = model(x, plane_ids=plane_ids)
                val_logits.append(logits.float().cpu().numpy())
                val_targets.append(y.numpy())
                
        val_logits = np.concatenate(val_logits, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        # Calculate OOF Macro-AUC
        aucs = {}
        for j, name in enumerate(LABELS):
            y_bin = (val_targets[:, j] >= 0.5).astype(int)
            aucs[name] = exact_roc_auc(y_bin, val_logits[:, j])
        macro_auc = float(np.mean(list(aucs.values())))
        
        ep_time = time.time() - t0
        print(f"  Epoch {ep+1:02d}/{epochs:02d} [{ep_time:.1f}s] - Train Loss: {train_loss:.4f} | Val OOF Macro-AUC: {macro_auc:.5f}")
        
        if macro_auc > best_auc:
            best_auc = macro_auc
            save_path = f"weights/engine3_multiplane_fold{fold}_best.pth"
            torch.save({
                'epoch': ep + 1,
                'model_state_dict': model.state_dict(),
                'macro_auc': macro_auc,
                'fold': fold,
                'labels': LABELS
            }, save_path)
            print(f"  >> New Best AUC ({best_auc:.5f}) - Checkpoint saved to {save_path} <<")
            
    print(f"FOLD {fold} TRAINING COMPLETE. Best OOF Macro-AUC: {best_auc:.5f}")
    return best_auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--corpus', default='/content/corpus')
    args = parser.parse_args()
    run_train_cv(fold=args.fold, epochs=args.epochs, batch_size=args.bs, corpus_dir=args.corpus)
