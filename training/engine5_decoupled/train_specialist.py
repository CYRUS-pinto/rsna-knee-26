"""
Training Engine 5: Plane-Decoupled Specialist Models (ADR-015).
Eliminates Multi-Task Representation Collapse via decoupled networks, differential LR,
frozen backbone warmup, and targeted loss functions.
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

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.plane_datasets import PlaneSpecialistDataset, SPECIALIST_TARGETS
from src.models.decoupled_specialists import SagittalSpecialistMIL, CoronalSpecialistMIL, AxialFluidSpecialistMIL
from src.models.losses import AsymmetricFocalLoss, JointSpaceOrdinalLoss, CalibratedSoftBCELoss

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

def get_specialist_model_and_loss(plane, backbone_name='convnext_small', pretrained=True):
    plane = plane.lower()
    if plane == 'sagittal':
        model = SagittalSpecialistMIL(backbone_name=backbone_name, pretrained=pretrained)
        # ACL, Medial Meniscus, Lateral Meniscus, Contusion
        loss_fn = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=2.0)
    elif plane == 'coronal':
        model = CoronalSpecialistMIL(backbone_name=backbone_name, pretrained=pretrained)
        # MCL, Medial OA, Lateral OA
        loss_fn = JointSpaceOrdinalLoss(margin_weight=0.1)
    elif plane == 'axial':
        model = AxialFluidSpecialistMIL(backbone_name=backbone_name, pretrained=pretrained)
        # PF OA, Effusion, Synovitis, Baker's, Fracture
        loss_fn = CalibratedSoftBCELoss()
    else:
        raise ValueError(f"Unknown plane {plane}")
    return model, loss_fn

def train_specialist_fold(plane='sagittal', fold=0, epochs=6, batch_size=2, accum_steps=8,
                          backbone_lr=1e-5, head_lr=1e-3, data_path='data', 
                          cached_volume_dir=None, backbone_name='convnext_small'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    targets = SPECIALIST_TARGETS[plane]
    
    print("=" * 75)
    print(f"  STARTING PLANE-DECOUPLED SPECIALIST TRAINING - [{plane.upper()}] FOLD {fold}")
    print(f"  Targets ({len(targets)}): {targets}")
    print(f"  Device: {device} | Epochs: {epochs} | Batch: {batch_size} (Accum: {accum_steps})")
    print(f"  Backbone LR: {backbone_lr} (Warmup Epoch 1 Frozen) | Head LR: {head_lr}")
    print("=" * 75)
    
    # 1. Load Labels & Folds
    folds_file = os.path.join(data_path, 'train_folds.parquet')
    if not os.path.exists(folds_file):
        folds_file = 'train_folds.parquet'
        
    labels_file = os.path.join(data_path, 'SUPER_CONSENSUS_labels.parquet')
    if not os.path.exists(labels_file):
        labels_file = 'SUPER_CONSENSUS_labels.parquet'
        
    df_labels = pd.read_parquet(labels_file) if labels_file.endswith('.parquet') else pd.read_csv(labels_file)
    from src.data.plane_datasets import ALL_LABELS
    labels_dict = {row['StudyInstanceUID']: row[ALL_LABELS].values.astype(np.float32) for _, row in df_labels.iterrows()}
    
    # Official Stratified Folds with Gold Isolation
    if os.path.exists(folds_file):
        df_folds = pd.read_parquet(folds_file)
        # Exclude Gold-58 studies strictly from train and val
        non_gold = df_folds[~df_folds['is_gold']]
        train_uids = non_gold[non_gold['fold'] != fold]['StudyInstanceUID'].tolist()
        val_uids_list = sorted(non_gold[non_gold['fold'] == fold]['StudyInstanceUID'].tolist())
        gold_uids_list = sorted(df_folds[df_folds['is_gold']]['StudyInstanceUID'].tolist())
        print(f"  [Official Folds]: Train={len(train_uids)} | Val={len(val_uids_list)} | Gold Holdout={len(gold_uids_list)}")
    else:
        # Fallback split
        all_uids = df_labels['StudyInstanceUID'].tolist()
        np.random.seed(42 + fold)
        shuffled_uids = list(all_uids)
        np.random.shuffle(shuffled_uids)
        val_count = len(shuffled_uids) // 5
        val_uids = set(shuffled_uids[fold * val_count : (fold + 1) * val_count])
        train_uids = [u for u in shuffled_uids if u not in val_uids]
        val_uids_list = sorted(list(val_uids))
        gold_uids_list = []
        print(f"  [Fallback Split]: Train={len(train_uids)} | Val={len(val_uids_list)}")
    
    # 2. Datasets & Loaders
    train_ds = PlaneSpecialistDataset(
        train_uids, labels_dict, plane=plane, 
        cached_volume_dir=cached_volume_dir, is_train=True
    )
    val_ds = PlaneSpecialistDataset(
        val_uids_list, labels_dict, plane=plane, 
        cached_volume_dir=cached_volume_dir, is_train=False
    )
    
    num_workers = 2 if os.name != 'nt' else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
                            
    # 3. Model & Loss Setup
    model, loss_fn = get_specialist_model_and_loss(plane, backbone_name=backbone_name)
    model = model.to(device)
    
    # Parameter groups for differential learning rates
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': backbone_lr},
        {'params': model.head.parameters(), 'lr': head_lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    
    best_macro_auc = 0.0
    best_oof_preds = None
    best_off_diag_corr = 1.0
    
    for epoch in range(epochs):
        t0 = time.time()
        
        # Warmup discipline: Freeze backbone during epoch 0 to stabilize attention heads
        if epoch == 0:
            for p in model.backbone.parameters():
                p.requires_grad = False
            print("  [Epoch 0 Warmup]: Backbone frozen. Optimizing gated attention heads only.")
        elif epoch == 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print("  [Epoch 1+ Full Fine-Tuning]: Backbone unfrozen at differential LR.")
            
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
        train_loss /= max(len(train_loader), 1)
        
        # Validation Loop
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        logits = model(x)
                else:
                    logits = model(x)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(y.numpy())
                
        val_preds = np.concatenate(val_preds, axis=0) if val_preds else np.zeros((len(val_uids_list), len(targets)))
        val_targets = np.concatenate(val_targets, axis=0) if val_targets else np.zeros((len(val_uids_list), len(targets)))
        
        # Class-Wise AUC
        class_aucs = {}
        for j, name in enumerate(targets):
            y_t = (val_targets[:, j] >= 0.5).astype(int)
            class_aucs[name] = exact_roc_auc(y_t, val_preds[:, j])
            
        macro_auc = float(np.mean(list(class_aucs.values())))
        
        # Representation Collapse Check: Off-Diagonal Correlation
        if val_preds.shape[1] > 1 and np.std(val_preds) > 1e-5:
            corr_matrix = np.corrcoef(val_preds.T)
            np.fill_diagonal(corr_matrix, 0.0)
            off_diag_corr = float(np.mean(np.abs(corr_matrix)))
        else:
            off_diag_corr = 0.0
            
        ep_time = time.time() - t0
        print(f"Epoch {epoch+1:02d}/{epochs:02d} [{ep_time:.1f}s] - Loss: {train_loss:.4f} | Macro-AUC: {macro_auc:.5f} | Mean Off-Diag Corr: {off_diag_corr:.4f}")
        for name, a in class_aucs.items():
            print(f"   - {name:20s}: {a:.4f}")
            
        # Model checkpointing
        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            best_oof_preds = val_preds
            best_off_diag_corr = off_diag_corr
            os.makedirs('weights', exist_ok=True)
            weight_path = f"weights/engine5_{plane}_fold{fold}_best.pth"
            torch.save(model.state_dict(), weight_path)
            print(f">> Saved New Best Specialist to {weight_path} (AUC: {macro_auc:.5f}, Corr: {off_diag_corr:.4f}) <<")
            
    # Save OOF Parquet
    if best_oof_preds is not None:
        os.makedirs('oof', exist_ok=True)
        df_oof = pd.DataFrame(best_oof_preds, columns=targets)
        df_oof['StudyInstanceUID'] = val_uids_list
        oof_path = f"oof/oof_engine5_{plane}_fold{fold}.parquet"
        df_oof.to_parquet(oof_path, index=False)
        print(f"Saved OOF predictions to {oof_path}")
        
    print("=" * 75)
    print(f"  FINISHED [{plane.upper()}] FOLD {fold} - Best AUC: {best_macro_auc:.5f} | Correlation: {best_off_diag_corr:.4f}")
    print("=" * 75)
    return best_macro_auc, best_off_diag_corr

def main():
    parser = argparse.ArgumentParser(description="Train Plane-Decoupled Specialist (ADR-015)")
    parser.add_argument("--plane", type=str, required=True, choices=['sagittal', 'coronal', 'axial'])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--backbone_name", type=str, default='convnext_small')
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--cached_volume_dir", type=str, default=None)
    args = parser.parse_args()
    
    train_specialist_fold(
        plane=args.plane,
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        backbone_name=args.backbone_name,
        data_path=args.data_path,
        cached_volume_dir=args.cached_volume_dir
    )

if __name__ == '__main__':
    main()
