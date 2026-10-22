"""
DINOv2-S/14 2.5D Multi-Slice Knee Pathology Classifier.
Implements:
1. Backbone: Vision Transformer (DINOv2-Small patch 14, 384 embedding dim).
2. 2.5D Slice Input: Ingests 3-slice anatomical packets [c-1, c, c+1].
3. Multi-Plane Sequence Pooling: Aggregates across Sagittal, Coronal, and Axial planes.
4. Learned Task Query Heads: 12 independent pathology query tokens with cross-attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class TaskQueryAttentionHead(nn.Module):
    """
    Learned Query Multi-Head Attention Head.
    12 independent learned target queries attend to spatial/temporal slice tokens.
    """
    def __init__(self, embed_dim=384, num_targets=12, num_heads=6, drop=0.1):
        super().__init__()
        self.num_targets = num_targets
        self.embed_dim = embed_dim
        
        # 12 Learned Query Tokens
        self.task_queries = nn.Parameter(torch.randn(num_targets, embed_dim) * 0.02)
        
        # Multi-Head Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True
        )
        self.norm_tokens = nn.LayerNorm(embed_dim)
        self.norm_queries = nn.LayerNorm(embed_dim)
        
        # Feed-forward classifier per target
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, token_seq, key_padding_mask=None):
        """
        Args:
            token_seq: (B, N_tokens, embed_dim)
            key_padding_mask: (B, N_tokens) bool tensor, True where padded
        Returns:
            logits: (B, 12)
        """
        B = token_seq.shape[0]
        tokens = self.norm_tokens(token_seq)
        
        # Expand queries to batch: (B, 12, embed_dim)
        q = self.norm_queries(self.task_queries).unsqueeze(0).expand(B, -1, -1)
        
        # Cross-attention: queries attend to visual tokens
        attn_out, _ = self.cross_attn(
            query=q,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Residual connection
        fused = q + attn_out  # (B, 12, embed_dim)
        
        # Compute logit per target: (B, 12)
        logits = self.mlp(fused).squeeze(-1)
        return logits

class DINOv2KneeClassifier(nn.Module):
    """
    Full 2.5D DINOv2 Knee Abnormality Model.
    Can be initialized with pretrained DINOv2 weights or dummy backbone for offline tests.
    """
    def __init__(self, backbone=None, embed_dim=384, num_targets=12, drop=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_targets = num_targets
        
        if backbone is not None:
            self.backbone = backbone
        else:
            # Fallback lightweight patch embedding for offline testing without torchhub download
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            
        self.head = TaskQueryAttentionHead(
            embed_dim=embed_dim,
            num_targets=num_targets,
            drop=drop
        )

    def extract_features(self, x):
        """
        x: (B * K, 3, H, W)
        Returns: (B * K, embed_dim)
        """
        if hasattr(self.backbone, 'forward_features'):
            # Standard timm or torchhub dinov2
            out = self.backbone.forward_features(x)
            if isinstance(out, dict):
                feats = out.get('x_norm_clstoken', out.get('cls_token'))
            elif hasattr(out, 'shape') and len(out.shape) == 3:
                feats = out[:, 0]  # CLS token
            else:
                feats = out
        else:
            feats = self.backbone(x)
        return feats

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, K_slices, 3, H, W) 2.5D slice packets
            mask: (B, K_slices) valid slice mask (1=valid, 0=padded)
        Returns:
            logits: (B, 12)
        """
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        
        feats = self.extract_features(flat_x)  # (B * K, embed_dim)
        feats = feats.view(B, K, self.embed_dim)
        
        key_padding_mask = (mask == 0) if mask is not None else None
        logits = self.head(feats, key_padding_mask=key_padding_mask)
        return logits
