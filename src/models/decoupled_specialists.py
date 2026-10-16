"""
ADR-015: Plane-Decoupled Specialist Architectures
Designed to eliminate Multi-Task Representation Collapse (0.9966 off-diagonal correlation).

Three Separate Networks:
1. SagittalSpecialistMIL: Focuses on ACL, Medial Meniscus, Lateral Meniscus, and Contusion.
2. CoronalSpecialistMIL: Focuses on MCL, Medial OA, and Lateral OA.
3. AxialFluidSpecialistMIL: Focuses on PF OA, Effusion, Synovitis, Baker's Cyst, and Fracture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class GatedAttentionPoolingHead(nn.Module):
    """
    Independent Gated Attention MIL mechanism for a specific subset of findings.
    Softmax attention across slices prevents dominant findings from overwriting focal tear signals.
    """
    def __init__(self, in_features, num_classes, hidden_dim=256, drop=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(in_features)
        
        # Gated Attention: tanh(V * h) * sigmoid(U * h)
        self.attention_V = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh(), nn.Dropout(drop))
            for _ in range(num_classes)
        ])
        self.attention_U = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid(), nn.Dropout(drop))
            for _ in range(num_classes)
        ])
        self.attention_weights = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_classes)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(in_features, 1) for _ in range(num_classes)
        ])
        
    def forward(self, feats):
        """
        Args:
            feats: (B, K, in_features)
        Returns:
            logits: (B, num_classes)
        """
        B, K, D = feats.shape
        h = self.norm(feats)
        
        logits_list = []
        for j in range(self.num_classes):
            v = self.attention_V[j](h) # (B, K, hidden)
            u = self.attention_U[j](h) # (B, K, hidden)
            scores = self.attention_weights[j](v * u).squeeze(-1) # (B, K)
            attn = torch.softmax(scores, dim=-1) # (B, K)
            
            # Weighted slice feature aggregation
            pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1) # (B, D)
            logit = self.classifiers[j](pooled) # (B, 1)
            logits_list.append(logit)
            
        return torch.cat(logits_list, dim=-1) # (B, num_classes)

class SagittalSpecialistMIL(nn.Module):
    """
    Specialist 1: Evaluates only Sagittal T2 & Sagittal T1 slices.
    Targets: ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Contusion'] (4 classes)
    """
    TARGETS = ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Contusion']
    
    def __init__(self, backbone_name='convnext_small', pretrained=True, in_chans=3):
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
        self.head = GatedAttentionPoolingHead(in_features=feat_dim, num_classes=len(self.TARGETS))
        
    def forward(self, x):
        """
        Args:
            x: (B, K_sag, 3, H, W)
        """
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        chunk_size = 16
        chunks = []
        for i in range(0, flat_x.size(0), chunk_size):
            chunks.append(self.backbone(flat_x[i:i + chunk_size]))
        feats = torch.cat(chunks, dim=0).view(B, K, -1)
        return self.head(feats)

class CoronalSpecialistMIL(nn.Module):
    """
    Specialist 2: Evaluates only Coronal T2 & Coronal T1 slices.
    Targets: ['MCL', 'Medial OA', 'Lateral OA'] (3 classes)
    """
    TARGETS = ['MCL', 'Medial OA', 'Lateral OA']
    
    def __init__(self, backbone_name='convnext_small', pretrained=True, in_chans=3):
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
        self.head = GatedAttentionPoolingHead(in_features=feat_dim, num_classes=len(self.TARGETS))
        
    def forward(self, x):
        """
        Args:
            x: (B, K_cor, 3, H, W)
        """
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        chunk_size = 16
        chunks = []
        for i in range(0, flat_x.size(0), chunk_size):
            chunks.append(self.backbone(flat_x[i:i + chunk_size]))
        feats = torch.cat(chunks, dim=0).view(B, K, -1)
        return self.head(feats)

class AxialFluidSpecialistMIL(nn.Module):
    """
    Specialist 3: Evaluates Axial slices and Fluid-Sensitive contrast.
    Targets: ['PF OA', 'Effusion', 'Synovitis', "Baker's", 'Fracture'] (5 classes)
    """
    TARGETS = ['PF OA', 'Effusion', 'Synovitis', "Baker's", 'Fracture']
    
    def __init__(self, backbone_name='convnext_small', pretrained=True, in_chans=3):
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
        self.head = GatedAttentionPoolingHead(in_features=feat_dim, num_classes=len(self.TARGETS))
        
    def forward(self, x):
        """
        Args:
            x: (B, K_ax, 3, H, W)
        """
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        chunk_size = 16
        chunks = []
        for i in range(0, flat_x.size(0), chunk_size):
            chunks.append(self.backbone(flat_x[i:i + chunk_size]))
        feats = torch.cat(chunks, dim=0).view(B, K, -1)
        return self.head(feats)
