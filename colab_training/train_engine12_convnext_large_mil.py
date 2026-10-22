"""
Engine 12: ConvNeXt-Large (198M params) with Gated Attention Multiple Instance Learning
and Asymmetric Focal Loss for RSNA Knee Abnormality Detection.
Designed for execution on Colab A100-80GB.
"""

import os, sys, time, gc
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

# 1. Hyperparameters & Configuration
MODEL_NAME = 'convnext_large.fb_in22k_ft_in1k_384'
FEAT_DIM = 1536
IMG_SIZE = 384
BATCH_SIZE = 8
GRAD_ACCUM = 2
LR = 1e-4
EPOCHS = 6
NUM_CLASSES = 12

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# 2. Asymmetric Focal Loss
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_pos = p
        p_neg = 1.0 - p
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)
        loss_pos = targets * torch.log(p_pos.clamp(min=self.eps)) * ((1.0 - p_pos) ** self.gamma_pos)
        loss_neg = (1.0 - targets) * torch.log(p_neg.clamp(min=self.eps)) * (p ** self.gamma_neg)
        return -(loss_pos + loss_neg).mean()

# 3. Model Architecture: ConvNeXt-Large + Gated Attention MIL Head
class ConvNeXtLargeMIL(nn.Module):
    def __init__(self, model_name=MODEL_NAME, feat_dim=FEAT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.norm = nn.LayerNorm(feat_dim)
        
        # Gated Attention Pooling
        self.attention_V = nn.Sequential(nn.Linear(feat_dim, 256), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(feat_dim, 256), nn.Sigmoid())
        self.attention_weights = nn.Linear(256, 1)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, S, C, H, W)
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W)
        h = self.backbone(x_flat)
        h = self.norm(h)
        h = h.view(B, S, -1)
        
        # Gated Attention weights
        V = self.attention_V(h)
        U = self.attention_U(h)
        A = self.attention_weights(V * U) # (B, S, 1)
        alpha = F.softmax(A, dim=1)
        
        # Volumetric representation
        vol_repr = torch.sum(h * alpha, dim=1) # (B, feat_dim)
        logits = self.classifier(vol_repr)
        return logits

def train_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = 0.0
    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, labels)
            loss = loss / GRAD_ACCUM
            
        scaler.scale(loss).backward()
        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * GRAD_ACCUM
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{len(loader)} | Running Loss: {total_loss / (step + 1):.4f}")
    return total_loss / len(loader)

if __name__ == '__main__':
    print("[+] ConvNeXt-Large MIL Engine 12 Architecture Loaded.")
