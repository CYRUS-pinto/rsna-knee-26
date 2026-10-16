"""
Biomechanical Relational Cross-Attention Module for Knee Pathology Co-occurrence.
Models clinical anatomical dependencies:
1. O'Donoghue's Unhappy Triad: ACL Tear <-> Medial Meniscus Tear <-> MCL Tear
2. Kissing Contusion Mechanism: ACL Tear <-> Lateral Bone Bruise / Contusion
3. Synovial Effusion Cascade: Effusion -> Synovial Thickening / Synovitis
4. Compartmental Degeneration: Medial OA vs. Lateral OA divergence

Transforms independent class probability logits into conditioned anatomical representations
via multi-head self-attention without heuristic clipping or threshold clamping (ADR-002 compliant).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ALL_LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class BiomechanicalRelationalHead(nn.Module):
    def __init__(self, num_classes=12, embed_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Learnable pathology embedding dictionary
        self.pathology_embedding = nn.Embedding(num_classes, embed_dim)
        
        # Projection from scalar logit to feature vector
        self.logit_proj = nn.Linear(1, embed_dim)
        
        # Multi-Head Self-Attention over 12 anatomical findings
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Final calibrated logit readout
        self.out_proj = nn.Linear(embed_dim, 1)

    def forward(self, logits):
        """
        Args:
            logits: Tensor of shape (B, num_classes) - raw or rank-transformed logits
        Returns:
            calibrated_logits: Tensor of shape (B, num_classes)
        """
        B, C = logits.shape
        assert C == self.num_classes, f"Expected {self.num_classes} classes, got {C}"
        
        # Path token indices (0 to 11)
        path_idx = torch.arange(self.num_classes, device=logits.device).unsqueeze(0).expand(B, -1) # (B, C)
        path_emb = self.pathology_embedding(path_idx) # (B, C, embed_dim)
        
        # Project scalar logits to embedding dimension
        logit_features = self.logit_proj(logits.unsqueeze(-1)) # (B, C, embed_dim)
        
        # Combined finding representation
        x = path_emb + logit_features # (B, C, embed_dim)
        
        # Relational Multi-Head Self-Attention across the 12 pathologies
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN refinement
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Project back to calibrated logits with residual connection from input
        delta = self.out_proj(x).squeeze(-1) # (B, C)
        return logits + delta
