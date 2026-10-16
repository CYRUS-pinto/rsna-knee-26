"""
Multi-Scale Spatial Pyramidal Transformer (MS-SPT) for Knee MRI.
Fuses Stream A (Native 512x512 Joint Margin Crops @ 0.35 mm/pixel)
with Stream B (Global 224x224 Whole Knee Context) via Bidirectional Cross-Attention.
Solves the fundamental trade-off: high-frequency ligament/cartilage fissures vs global knee alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

ALL_LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class CrossAttentionFusionBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha_local_to_global = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_global_to_local = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, local_feats, global_feats):
        # local_feats: (B, K, embed_dim) - 512 joint crop sequence
        # global_feats: (B, K, embed_dim) - 224 whole knee sequence
        
        # Local queries global context
        local_attended, _ = self.mha_local_to_global(local_feats, global_feats, global_feats)
        local_res = self.norm1(local_feats + local_attended)
        
        # Global queries local details
        global_attended, _ = self.mha_global_to_local(global_feats, local_feats, local_feats)
        global_res = self.norm2(global_feats + global_attended)
        
        # Spliced multimodal representation
        combined = torch.cat([local_res, global_res], dim=-1)
        out = self.norm_out(local_res + self.ffn(combined))
        return out

class MultiScalePyramidalMIL(nn.Module):
    def __init__(self, num_classes=12, embed_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes
        
        # Stream A: Local 512 Joint Space Backbone
        self.local_backbone = timm.create_model('resnet34d', pretrained=True, in_chans=3, num_classes=0)
        local_dim = self.local_backbone.num_features
        self.local_proj = nn.Linear(local_dim, embed_dim)
        
        # Stream B: Global 224 Whole Knee Backbone
        self.global_backbone = timm.create_model('convnext_nano', pretrained=True, in_chans=3, num_classes=0)
        global_dim = self.global_backbone.num_features
        self.global_proj = nn.Linear(global_dim, embed_dim)
        
        # Bidirectional Cross-Attention
        self.fusion = CrossAttentionFusionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        # Sequential Inter-Slice BiLSTM
        self.bilstm = nn.LSTM(embed_dim, embed_dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        
        # Class-Specific Gated Attention Readout
        self.attention_v = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, 128), nn.Tanh()) for _ in range(num_classes)])
        self.attention_u = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, 128), nn.Sigmoid()) for _ in range(num_classes)])
        self.attention_w = nn.ModuleList([nn.Linear(128, 1) for _ in range(num_classes)])
        self.classifiers = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_classes)])

    def forward(self, local_x, global_x):
        """
        Args:
            local_x: (B, K, 3, 512, 512) - Sequence of native 512 joint line slabs
            global_x: (B, K, 3, 224, 224) - Sequence of global 224 whole knee slabs
        Returns:
            logits: (B, num_classes)
        """
        B, K, C, Hl, Wl = local_x.shape
        _, _, _, Hg, Wg = global_x.shape
        
        # Feature extraction
        loc_flat = local_x.view(B * K, C, Hl, Wl)
        loc_feats = self.local_proj(self.local_backbone(loc_flat)).view(B, K, -1)
        
        glob_flat = global_x.view(B * K, C, Hg, Wg)
        glob_feats = self.global_proj(self.global_backbone(glob_flat)).view(B, K, -1)
        
        # Pyramidal Cross-Attention Fusion
        fused = self.fusion(loc_feats, glob_feats) # (B, K, embed_dim)
        
        # Sequence temporal modeling
        lstm_out, _ = self.bilstm(fused) # (B, K, embed_dim)
        
        # Gated MIL Pooling per class
        logits = []
        for c in range(self.num_classes):
            v = self.attention_v[c](lstm_out)
            u = self.attention_u[c](lstm_out)
            a = self.attention_w[c](v * u).squeeze(-1) # (B, K)
            alpha = F.softmax(a, dim=1).unsqueeze(-1)
            pooled = torch.sum(lstm_out * alpha, dim=1) # (B, embed_dim)
            logits.append(self.classifiers[c](pooled))
            
        return torch.cat(logits, dim=-1)
