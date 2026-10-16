"""
Sequence-Aware 2.5D Volumetric MIL Model with Bidirectional LSTM for Inter-Slice Continuity.
Addresses Pillar 2 of ADR-013 & Karpathy simplicity guidelines:
- In volumetric MRI, ligaments (ACL, MCL) and meniscal horns are continuous structures spanning multiple slices.
- Standard MIL treats slices as an unordered bag; this model processes slice representations as an ordered sequence.
- Features: 
  1. 2.5D slice triplet extraction (c-1, c, c+1).
  2. Spatial backbone (timm) extracts slice-level features.
  3. 2-layer Bidirectional LSTM models inter-slice anatomical continuity.
  4. Gated Attention pooling computes finding-specific volume-level logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SequenceGatedAttentionPooling(nn.Module):
    """
    Finding-specific gated attention pooling across sequence slices.
    Combines temporal sequence features into study-level abnormality logits.
    """
    def __init__(self, in_features, num_classes, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(in_features)
        
        self.attn_v = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh(), nn.Dropout(dropout))
            for _ in range(num_classes)
        ])
        self.attn_u = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout))
            for _ in range(num_classes)
        ])
        self.attn_w = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_classes)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(in_features, 1) for _ in range(num_classes)
        ])
        
    def forward(self, seq_feats):
        """
        Args:
            seq_feats: (B, K, in_features) sequence of slice features
        Returns:
            logits: (B, num_classes)
        """
        B, K, D = seq_feats.shape
        h = self.norm(seq_feats)
        
        logits = []
        for j in range(self.num_classes):
            v = self.attn_v[j](h)
            u = self.attn_u[j](h)
            scores = self.attn_w[j](v * u).squeeze(-1) # (B, K)
            weights = torch.softmax(scores, dim=-1) # (B, K)
            
            pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(1) # (B, D)
            logits.append(self.classifiers[j](pooled))
            
        return torch.cat(logits, dim=-1)

class VolumetricSequenceMIL(nn.Module):
    def __init__(
        self,
        num_classes=12,
        backbone_name='convnext_small',
        pretrained=True,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.2,
        in_chans=3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            global_pool='avg'
        )
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            try:
                self.backbone.set_grad_checkpointing(True)
            except Exception:
                pass
                
        feat_dim = self.backbone.num_features
        
        # Bidirectional LSTM to model volumetric inter-slice continuity
        self.bilstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # Projection with residual connection: input feat_dim + (2 * lstm_hidden) -> feat_dim
        lstm_out_dim = lstm_hidden * 2
        self.proj = nn.Sequential(
            nn.Linear(feat_dim + lstm_out_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.pool = SequenceGatedAttentionPooling(feat_dim, num_classes, hidden_dim=256, dropout=dropout)

    def forward(self, x):
        """
        Args:
            x: (B, K, 3, H, W) where K is the ordered slice sequence along the normal axis
        Returns:
            logits: (B, num_classes)
        """
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        
        # Process in chunks to prevent VRAM spikes
        chunk_size = 16
        feats_list = []
        for i in range(0, flat_x.size(0), chunk_size):
            feats_list.append(self.backbone(flat_x[i:i + chunk_size]))
        spatial_feats = torch.cat(feats_list, dim=0).view(B, K, -1) # (B, K, feat_dim)
        
        # Model slice-to-slice temporal continuity
        lstm_out, _ = self.bilstm(spatial_feats) # (B, K, 2 * lstm_hidden)
        
        # Residual fusion of spatial identity and volumetric context
        fused = self.proj(torch.cat([spatial_feats, lstm_out], dim=-1)) # (B, K, feat_dim)
        
        return self.pool(fused)
