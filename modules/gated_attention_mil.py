"""
Gated Attention Multiple Instance Learning (MIL) Module
Reference: Ilse, Tomczak, Welling (ICML 2018) "Attention-based Deep Multiple Instance Learning"
Adapted for Multi-Slice Knee MRI Feature Pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttentionMIL(nn.Module):
    """
    Gated Attention Pooling Layer for Bag of Slices.
    Maps a sequence of slice embeddings (B, S, D) to a single study embedding (B, D).
    Computes attention weights:
        a_k = softmax(w^T (tanh(V h_k) * sigmoid(U h_k)))
    """
    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        
        self.attention_V = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, return_attention: bool = False):
        """
        Args:
            h: Tensor of shape (B, S, D) where S is number of slices, D is feature dim.
        Returns:
            pooled: Tensor of shape (B, D)
            alpha (optional): Attention weights (B, S, 1)
        """
        # h: (B, S, D)
        V = self.attention_V(h)  # (B, S, hidden_dim)
        U = self.attention_U(h)  # (B, S, hidden_dim)
        
        # Element-wise gating
        gated = V * U
        gated = self.dropout(gated)
        
        # Attention scores
        A = self.attention_weights(gated)  # (B, S, 1)
        alpha = F.softmax(A, dim=1)        # (B, S, 1)
        
        # Weighted sum of slice representations
        pooled = torch.sum(h * alpha, dim=1)  # (B, D)
        
        if return_attention:
            return pooled, alpha
        return pooled

class MultiPathologyMILClassifier(nn.Module):
    """
    Complete MIL Backbone with separate Gated Attention per pathology or shared with multi-head.
    """
    def __init__(self, backbone_dim: int = 1024, num_classes: int = 12, hidden_dim: int = 256):
        super().__init__()
        self.pooler = GatedAttentionMIL(in_features=backbone_dim, hidden_dim=hidden_dim)
        self.norm = nn.LayerNorm(backbone_dim)
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, slice_feats):
        """
        Args:
            slice_feats: (B, S, D)
        Returns:
            logits: (B, num_classes)
        """
        pooled = self.pooler(slice_feats)  # (B, D)
        pooled = self.norm(pooled)
        logits = self.classifier(pooled)
        return logits

if __name__ == '__main__':
    # Unit test
    x = torch.randn(2, 40, 1024)
    model = MultiPathologyMILClassifier(backbone_dim=1024, num_classes=12)
    out = model(x)
    assert out.shape == (2, 12), f"Expected (2, 12), got {out.shape}"
    print("[+] GatedAttentionMIL unit test passed! Output shape:", out.shape)
