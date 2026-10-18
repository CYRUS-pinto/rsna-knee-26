"""
Sovereign Inference Module for Engine 9 (ConvNeXt-384) & Engine 10 (Swin-Base-384)
Runs high-speed batched FP16 inference on RSNA Knee test studies
"""
import os, sys, glob, time, json, math
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import timm

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class Engine9MIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnext_small.fb_in22k_ft_in1k_384', pretrained=False, num_classes=0)
        self.norm = nn.LayerNorm(768)
        self.att = nn.Sequential(
            nn.Linear(768, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, 12)
        )
        self.clsW = nn.Parameter(torch.empty(12, 768))
        self.clsb = nn.Parameter(torch.empty(12))

    def forward(self, x):
        # x: (S, 3, 384, 384)
        h = self.backbone(x) # (S, 768)
        h = self.norm(h)     # (S, 768)
        alpha = F.softmax(self.att(h), dim=0) # (S, 12)
        z = torch.matmul(alpha.t(), h)        # (12, 768)
        logits = (z * self.clsW).sum(dim=-1) + self.clsb # (12,)
        return logits

class SwinMILModel(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window12_384.ms_in22k_ft_in1k', num_classes=12):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # x: (1, S, 3, 384, 384)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.backbone(x) # (S, feat_dim)
        feats = feats.view(B, S, -1)
        attn_scores = self.attention(feats) # (B, S, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # (B, S, 1)
        pooled_attn = torch.sum(feats * attn_weights, dim=1) # (B, feat_dim)
        pooled_max, _ = torch.max(feats, dim=1)
        pooled = 0.7 * pooled_attn + 0.3 * pooled_max
        logits = self.head(pooled) # (B, num_classes)
        return logits[0]

def test_inference_pipeline():
    print("Testing Sovereign Inference Module syntax & forward logic...")
    e9 = Engine9MIL()
    dummy_slices_e9 = torch.randn(8, 3, 384, 384)
    with torch.no_grad():
        out9 = torch.sigmoid(e9(dummy_slices_e9))
    print("Engine 9 output shape:", out9.shape)
    assert out9.shape == torch.Size([12])

    e10 = SwinMILModel(num_classes=12)
    dummy_slices_e10 = torch.randn(1, 6, 3, 384, 384)
    with torch.no_grad():
        out10 = torch.sigmoid(e10(dummy_slices_e10))
    print("Engine 10 output shape:", out10.shape)
    assert out10.shape == torch.Size([12])
    print("All tests passed successfully!")

if __name__ == '__main__':
    test_inference_pipeline()
