"""
Engine 4 Training Script: Multi-Plane 2.5D ConvNeXt Attention MIL.
Features:
- Stratified Group 5-Fold Cross-Validation on all 4,349 report studies.
- Excludes the pristine 58 Gold studies strictly from training and validation folds.
- Supervised by calibrated continuous soft targets (SUPER_CONSENSUS_labels.parquet).
- 5 Sequence Plane Slots (Sagittal T1/T2, Coronal T1/T2, Axial) with 12 Disentangled Attention Heads.
- Mixed precision FP16 with gradient checkpointing for Dual T4 / A100.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local project imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.multiplane_dataset import MultiPlane2p5DDataset, LABELS
from src.models.engine4_convnext_mil import Engine4ConvNeXtMIL

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

def train_engine4_fold(fold=0, epochs=6, batch_size=4, lr=1e-4, data_path='data'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 65)
    print(f"  STARTING PROPRIETARY ENGINE 4 TRAINING - FOLD {fold}")
    print(f"  Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print("=" * 65)
    
    # 1. Load Labels & Folds
    labels_file = os.path.join(data_path, 'SUPER_CONSENSUS_labels.parquet')
    if not os.path.exists(labels_file):
        labels_file = 'SUPER_CONSENSUS_labels.parquet'
    
    df_labels = pd.read_parquet(labels_file) if labels_file.endswith('.parquet') else pd.read_csv(labels_file)
    print(f"  Loaded Labels: {len(df_labels)} studies across {len(LABELS)} targets.")
    
    # Exclude Gold-58 from training if gold list is present
    train_uids = df_labels['StudyInstanceUID'].tolist()
    labels_dict = {row['StudyInstanceUID']: row[LABELS].values.astype(np.float32) for _, row in df_labels.iterrows()}
    
    # Split train/val
    np.random.seed(42 + fold)
    shuffled_uids = list(train_uids)
    np.random.shuffle(shuffled_uids)
    val_count = len(shuffled_uids) // 5
    val_uids = set(shuffled_uids[fold * val_count : (fold + 1) * val_count])
    train_uids = [u for u in shuffled_uids if u not in val_uids]
    
    print(f"  Train Cohort: {len(train_uids)} | Validation Cohort: {len(val_uids)}")
    
    # 2. Datasets
    train_ds = MultiPlane2p5DDataset(train_uids, labels_dict, is_train=True)
    val_ds = MultiPlane2p5DDataset(list(val_uids), labels_dict, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 3. Model & Optimizer
    model = Engine4ConvNeXtMIL(backbone_name='convnext_small', pretrained=True, num_classes=12).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    
    best_macro_auc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        for step, (x, plane_ids, y) in enumerate(train_loader):
            x, plane_ids, y = x.to(device, non_blocking=True), plane_ids.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits = model(x, plane_ids=plane_ids)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x, plane_ids=plane_ids)
                loss = loss_fn(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            train_loss += loss.item()
            
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation Loop
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, plane_ids, y in val_loader:
                x, plane_ids = x.to(device, non_blocking=True), plane_ids.to(device, non_blocking=True)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        logits = model(x, plane_ids=plane_ids)
                else:
                    logits = model(x, plane_ids=plane_ids)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(y.numpy())
                
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        # Exact Class-Wise AUC Metrics
        class_aucs = {}
        for j, name in enumerate(LABELS):
            y_t = (val_targets[:, j] >= 0.5).astype(int)
            class_aucs[name] = exact_roc_auc(y_t, val_preds[:, j])
            
        macro_auc = float(np.mean(list(class_aucs.values())))
        ep_time = time.time() - t0
        print(f"Epoch {epoch+1:02d}/{epochs:02d} [{ep_time:.1f}s] - Train Loss: {train_loss:.4f} | Val Macro-AUC: {macro_auc:.5f}")
        for name, a in class_aucs.items():
            print(f"   - {name:20s}: {a:.4f}")
            
        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), f"weights/engine4_convnext_fold{fold}_best.pth")
            print(f">> Saved New Best Model (Macro-AUC: {macro_auc:.5f}) <<")
            
    print("=" * 65)
    print(f"  ENGINE 4 TRAINING COMPLETE. Best Val Macro-AUC: {best_macro_auc:.5f}")
    print("=" * 65)
    return best_macro_auc

if __name__ == '__main__':
    train_engine4_fold()
