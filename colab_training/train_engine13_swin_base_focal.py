"""
Engine 13: Swin-Base-384 with Asymmetric Focal Loss and Multi-Scale Attention
Targeting High-Fidelity Cruciate & Meniscal Cartilage Detection.
Designed for Colab A100.
"""

import os, sys, time, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import timm

MODEL_NAME = 'swin_base_patch4_window12_384.ms_in22k_ft_in1k'
IMG_SIZE = 384
NUM_CLASSES = 12

class SwinFocalMIL(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W)
        feats = self.backbone(x_flat)
        feats = feats.view(B, S, -1)
        
        attn_scores = self.attention(feats)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        pooled_attn = torch.sum(feats * attn_weights, dim=1)
        pooled_max, _ = torch.max(feats, dim=1)
        pooled = 0.7 * pooled_attn + 0.3 * pooled_max
        logits = self.head(pooled)
        return logits

if __name__ == '__main__':
    print("[+] Swin-Base-384 Focal MIL Engine 13 Architecture Loaded.")
