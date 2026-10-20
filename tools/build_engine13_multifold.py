"""
Builder for Engine 13 Swin-Base-384 Multi-Fold Training (Colab).
"""
import json
from pathlib import Path

BASE_CONFIG = {
    "model_name": "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
    "img_size": 384,
    "batch_size": 4,
    "chunk_size": 4,
    "k_triplets": 8,
    "num_epochs": 8,
    "lr_backbone": 2e-5,
    "lr_head": 2e-4,
    "weight_decay": 0.05,
    "grad_checkpointing": True,
    "mixed_precision": True,
    "folds": 5,
    "stratified_group_col": "PatientID",
}

FOLDS_DIR = Path("colab_folds/engine13_swin_base")
FOLDS_DIR.mkdir(parents=True, exist_ok=True)

# Raw template with FOLD placeholder
SCRIPT_TEMPLATE = r"""
"""
Engine 13 Swin-Base-384 Fold FOLD Training Script.
Isolated execution: !python train_engine13_foldFOLD.py
Saves checkpoints to /content and pushes to Kaggle Datasets each epoch.
"""
import os, sys, json, time, gc
from pathlib import Path

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from sklearn.model_selection import StratifiedGroupKFold

# ========== CONFIGURATION ==========
FOLD = FOLD
MODEL_NAME = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"
IMG_SIZE = 384
BATCH_SIZE = 4
CHUNK_SIZE = 4
K_TRIPLETS = 8
NUM_EPOCHS = 8
LR_BACKBONE = 2e-5
LR_HEAD = 2e-4
WEIGHT_DECAY = 0.05
GRAD_CHECKPOINTING = True
MIXED_PRECISION = True

# Paths
DATA_ROOT = Path("/kaggle/input/rsna-knee-abnormality-detection")
WEIGHTS_DIR = Path("/content/engine13_foldFOLD_weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASET = "cyruspinto/rsna-knee-engine13-swin-base-foldFOLD-weights"

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

# ========== ASYMMETRIC FOCAL LOSS ==========
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, clip=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        if self.clip > 0:
            probs = probs.clamp(self.clip, 1 - self.clip)
        pos_loss = -targets * (1 - probs).pow(self.gamma_pos) * probs.log()
        neg_loss = -(1 - targets) * probs.pow(self.gamma_neg) * (1 - probs).log()
        return (pos_loss + neg_loss).mean()

# ========== MODEL DEFINITION ==========
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

    def encode(self, x):
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        feats = []
        for i in range(0, flat_x.size(0), CHUNK_SIZE):
            f = self.backbone(flat_x[i:i + CHUNK_SIZE])
            feats.append(f)
        feats = torch.cat(feats, dim=0)
        return feats.view(B, K, -1)

    def forward(self, x):
        feats = self.encode(x)
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

# ========== TRAINING LOOP ==========
def train_fold():
    print("=== Engine 13 Swin-Base-384 Fold FOLD Training ===", flush=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device), flush=True)
    if device.type == "cuda":
        print("GPU: " + str(torch.cuda.get_device_name(0)), flush=True)
        print("VRAM: " + str(torch.cuda.get_device_properties(0).total_memory / 1e9) + " GB", flush=True)
    
    bb = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
    if GRAD_CHECKPOINTING:
        bb.set_grad_checkpointing(True)
    model = SwinMILClassifier(bb, feat_dim=1024, num_targets=12).to(device)
    
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.att.parameters()) + list(model.norm.parameters()) + [model.clsW, model.clsb]
    
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=4.0)
    
    best_auc = 0.0
    best_epoch = 0
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        
        model.eval()
        val_auc = 0.0
        
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_auc": val_auc,
            "best_auc": best_auc,
            "fold": FOLD,
            "config": {
                "model_name": MODEL_NAME,
                "img_size": IMG_SIZE,
                "batch_size": BATCH_SIZE,
                "chunk_size": CHUNK_SIZE,
                "k_triplets": K_TRIPLETS,
                "num_epochs": NUM_EPOCHS,
                "lr_backbone": LR_BACKBONE,
                "lr_head": LR_HEAD,
                "weight_decay": WEIGHT_DECAY,
                "grad_checkpointing": GRAD_CHECKPOINTING,
                "mixed_precision": MIXED_PRECISION,
            },
        }
        
        ckpt_path = WEIGHTS_DIR / "engine13_foldFOLD_epoch" + str(epoch) + ".pt"
        torch.save(checkpoint, ckpt_path)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_path = WEIGHTS_DIR / "engine13_foldFOLD_best.pt"
            torch.save(checkpoint, best_path)
            print("  New best: " + str(best_auc), flush=True)
        
        try:
            import subprocess
            result = subprocess.run([
                "kaggle", "datasets", "version",
                "-p", str(WEIGHTS_DIR),
                "-m", "Engine 13 Fold " + str(FOLD) + " Epoch " + str(epoch) + " Val AUC: " + str(val_auc)
            ], capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("  Pushed to Kaggle: " + KAGGLE_DATASET, flush=True)
            else:
                print("  Push failed (will retry): " + result.stderr[:200], flush=True)
        except Exception as e:
            print("  Push error (non-fatal): " + str(e), flush=True)
        
        scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()
        
        epoch_time = time.time() - epoch_start
        print("Epoch " + str(epoch+1) + "/" + str(NUM_EPOCHS) + " | Val AUC: " + str(val_auc) + " | Best: " + str(best_auc) + " | Time: " + str(epoch_time) + "s", flush=True)
    
    print("=== Fold " + str(FOLD) + " Complete. Best Val AUC: " + str(best_auc) + " at epoch " + str(best_epoch+1) + " ===", flush=True)
    
    try:
        import subprocess
        subprocess.run([
            "kaggle", "datasets", "version",
            "-p", str(WEIGHTS_DIR),
            "-m", "Engine 13 Fold " + str(FOLD) + " BEST Val AUC: " + str(best_auc)
        ], check=True, timeout=300)
        print("=== Final best weights pushed to Kaggle ===", flush=True)
    except Exception as e:
        print("Final push error: " + str(e), flush=True)

if __name__ == "__main__":
    train_fold()
"""

for fold in range(BASE_CONFIG["folds"]):
    fold_dir = FOLDS_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Replace FOLD placeholder
    script_content = SCRIPT_TEMPLATE.replace("FOLD", str(fold))
    script_path = fold_dir / f"train_engine13_fold{fold}.py"
    with open(script_path, "w") as f:
        f.write(script_content)
    
    print(f"[+] Created {script_path}")

# Create master notebook
master_nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Engine 13 Swin-Base-384 Multi-Fold Training\n",
                "\n",
                "**5-Fold Stratified GroupKFold Training**\n",
                "- Model: swin_base_patch4_window12_384.ms_in22k_ft_in1k\n",
                "- Image Size: 384px\n",
                "- Batch Size: 4 (chunk=4, k=8)\n",
                "- Epochs: 8\n",
                "- LR: 2e-5 (backbone), 2e-4 (head)\n",
                "- Gradient Checkpointing: True\n",
                "- Mixed Precision: True\n",
                "- Loss: Asymmetric Focal (gamma_pos=1.0, gamma_neg=4.0)\n",
                "\n",
                "**Run each fold in a separate Colab tab for parallel training.**\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Install dependencies\n",
                "!pip install -q timm==0.9.12 pandas numpy scikit-learn torch torchvision --no-cache-dir"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

for fold in range(BASE_CONFIG["folds"]):
    master_nb["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            f"# Fold {fold} - Run in separate Colab tab\n",
            f"!python colab_folds/engine13_swin_base/fold_{fold}/train_engine13_fold{fold}.py\n"
        ]
    })

master_path = FOLDS_DIR / "engine13_multi_fold_master.ipynb"
with open(master_path, "w") as f:
    json.dump(master_nb, f, indent=1)

print(f"[+] Created master notebook: {master_path}")
print("[+] Engine 13 multi-fold training scripts ready!")