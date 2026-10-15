"""
RSNA Knee Abnormality Detection - Multi-Plane Slice Transformer (Engine 3 Aligned)
Architecture:
1. Multi-Sequence / Multi-Plane Aware Positional Encoding (ADR-004).
   Solves the 44-slice (training) vs 96-slice multi-plane (inference) spatial disconnect.
   Combines intra-sequence continuous 1D depth embedding with discrete sequence/plane IDs.
2. Cross-Sequence Multi-Head Self-Attention (Transformer Encoder).
3. 12-Class Condition-Specific Attention-MIL Pooling.
4. Direct raw logit output for ROC AUC numerical stability (ADR-003).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class MultiPlaneSliceTransformerHead(nn.Module):
    """
    Multi-Plane Attention-MIL Transformer Head.
    Supports variable slice counts K in [16, 128] and variable plane counts.
    """
    def __init__(self, F_dim=768, n=12, num_heads=8, num_layers=2, max_slices=128, max_planes=8, drop=0.1):
        super().__init__()
        self.F_dim = F_dim
        self.n = n
        self.norm = nn.LayerNorm(F_dim)
        
        # Intra-plane relative depth embedding (continuous depth across a given sequence)
        self.intra_pos_emb = nn.Parameter(torch.zeros(1, max_slices, F_dim))
        nn.init.trunc_normal_(self.intra_pos_emb, std=0.02)
        
        # Discrete plane sequence embedding (0: Sagittal, 1: Coronal, 2: Axial, etc.)
        self.plane_emb = nn.Embedding(max_planes, F_dim)
        nn.init.trunc_normal_(self.plane_emb.weight, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=F_dim, nhead=num_heads, dim_feedforward=F_dim * 2,
            dropout=drop, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Class-specific attention pooling weights
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
        
    def forward(self, feats, plane_ids=None):
        """
        feats: Tensor of shape (B, K, F_dim)
        plane_ids: Optional LongTensor of shape (B, K) or (K,) indicating plane index per slice
        Returns: logits of shape (B, n)
        """
        B, K, D = feats.shape
        h = self.norm(feats)
        
        # Compute positional embeddings
        pos = self.intra_pos_emb[:, :K, :]
        if plane_ids is not None:
            if plane_ids.dim() == 1:
                p_emb = self.plane_emb(plane_ids).unsqueeze(0) # (1, K, D)
            else:
                p_emb = self.plane_emb(plane_ids) # (B, K, D)
            h = h + pos + p_emb
        else:
            h = h + pos
            
        h_trans = self.transformer(h)
        
        # Attention pooling: softmax over slices (dim=1) for each class
        # att: (B, K, n)
        a = torch.softmax(self.att(h_trans), dim=1)
        
        # Weighted aggregate per class: (B, n, D)
        pooled = torch.einsum("bkn,bkf->bnf", a, h_trans)
        pooled = self.post_norm(pooled)
        
        # Classification projection
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

class MultiPlaneKneeModel(nn.Module):
    """
    End-to-End Multi-Plane Model wrapper.
    Accepts 2D feature extractor backbone + MultiPlaneSliceTransformerHead.
    """
    def __init__(self, backbone=None, F_dim=768, n=12, num_heads=8, num_layers=2, drop=0.1):
        super().__init__()
        self.backbone = backbone
        self.head = MultiPlaneSliceTransformerHead(
            F_dim=F_dim, n=n, num_heads=num_heads, num_layers=num_layers, drop=drop
        )
        
    def encode(self, x):
        """
        x: Tensor of shape (B, K, C, H, W)
        Returns: features of shape (B, K, F_dim)
        """
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        chunk_size = 16
        chunks = []
        for i in range(0, flat_x.size(0), chunk_size):
            chunk = flat_x[i:i + chunk_size]
            chunks.append(self.backbone(chunk))
        feats = torch.cat(chunks, dim=0)
        return feats.view(B, K, -1)
        
    def forward(self, x, plane_ids=None):
        if self.backbone is not None:
            feats = self.encode(x)
        else:
            # If backbone is None, x is treated as precomputed features (B, K, F_dim)
            feats = x
        return self.head(feats, plane_ids=plane_ids)
