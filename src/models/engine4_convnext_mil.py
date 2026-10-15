"""
Engine 4 Architecture: ConvNeXt-Small Multi-Plane Multiple Instance Learning (MIL)
Key Features:
1. Backbone: timm convnext_small with gradient checkpointing.
2. 5-Slot Learnable Sequence Plane Embeddings (Sagittal T1/T2, Coronal T1/T2, Axial).
3. 12 Disentangled Per-Finding Attention MIL Heads (one attention vector per pathology).
4. Direct output of 12 unconstrained logits for stable BCEWithLogitsLoss optimization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class PerFindingAttentionMILHead(nn.Module):
    """
    12 Independent Attention MIL heads.
    Rather than forcing all pathologies to pool slices identically,
    each pathology computes its own attention distribution across the K slice windows.
    """
    def __init__(self, in_features, num_classes=12, hidden_dim=256, num_planes=5, drop=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        
        # Sequence Plane Embeddings
        self.plane_emb = nn.Embedding(num_planes, in_features)
        nn.init.trunc_normal_(self.plane_emb.weight, std=0.02)
        
        # Input LayerNorm
        self.norm = nn.LayerNorm(in_features)
        
        # 12 Independent Attention Mechanisms: V and U branches (Gated Attention)
        self.attention_V = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.Tanh(),
                nn.Dropout(drop)
            ) for _ in range(num_classes)
        ])
        self.attention_U = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.Sigmoid(),
                nn.Dropout(drop)
            ) for _ in range(num_classes)
        ])
        self.attention_weights = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_classes)
        ])
        
        # 12 Independent Classifiers
        self.classifiers = nn.ModuleList([
            nn.Linear(in_features, 1) for _ in range(num_classes)
        ])
        
    def forward(self, feats, plane_ids=None):
        """
        Args:
            feats: (B, K, in_features)
            plane_ids: (B, K) or (K,) long tensor of plane slot indices.
        Returns:
            logits: (B, 12)
        """
        B, K, D = feats.shape
        h = self.norm(feats)
        
        # Add discrete plane slot embeddings if provided
        if plane_ids is not None:
            if plane_ids.dim() == 1:
                p_emb = self.plane_emb(plane_ids).unsqueeze(0).expand(B, -1, -1)
            else:
                p_emb = self.plane_emb(plane_ids)
            h = h + p_emb
            
        logits_list = []
        for j in range(self.num_classes):
            # Gated Attention: A_j = softmax(w_j^T (tanh(V_j h) * sigmoid(U_j h)))
            v = self.attention_V[j](h) # (B, K, hidden)
            u = self.attention_U[j](h) # (B, K, hidden)
            scores = self.attention_weights[j](v * u).squeeze(-1) # (B, K)
            attn = torch.softmax(scores, dim=-1) # (B, K)
            
            # Weighted slice feature aggregation
            # (B, 1, K) x (B, K, D) -> (B, D)
            pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1) # (B, D)
            
            # Linear classification
            logit = self.classifiers[j](pooled) # (B, 1)
            logits_list.append(logit)
            
        logits = torch.cat(logits_list, dim=-1) # (B, 12)
        return logits

class Engine4ConvNeXtMIL(nn.Module):
    def __init__(self, backbone_name='convnext_small', pretrained=True, in_chans=3, 
                 num_classes=12, num_planes=5, hidden_dim=256, drop=0.1):
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
        self.head = PerFindingAttentionMILHead(
            in_features=feat_dim, num_classes=num_classes, 
            hidden_dim=hidden_dim, num_planes=num_planes, drop=drop
        )
        
    def forward(self, x, plane_ids=None):
        """
        Args:
            x: (B, K, 3, H, W) 2.5D slice triplet tensor.
            plane_ids: (B, K) or (K,) long tensor indicating sequence slot.
        Returns:
            logits: (B, 12)
        """
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        
        # Chunked forward pass through backbone to prevent GPU VRAM exhaustion
        chunk_size = 16
        chunks = []
        for i in range(0, flat_x.size(0), chunk_size):
            chunks.append(self.backbone(flat_x[i:i + chunk_size]))
        feats = torch.cat(chunks, dim=0).view(B, K, -1)
        
        return self.head(feats, plane_ids=plane_ids)
