"""
Loss Functions for Plane-Decoupled Specialists (ADR-015).
Includes:
1. AsymmetricFocalLoss: Focuses on hard focal acute tears (ACL, MCL, Menisci), suppresses easy negatives.
2. CalibratedSoftBCELoss: Continuous report target BCE with label smoothing.
3. JointSpaceOrdinalLoss: Enforces ordinal severity constraints on joint space narrowing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for highly imbalanced multi-label medical imaging.
    Penalizes false negatives on rare acute tears more heavily than easy negatives.
    """
    def __init__(self, gamma_pos=1.0, gamma_neg=2.0, clip=0.05, pos_weight=None):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) unnormalized log-odds
            targets: (B, C) continuous or binary in [0, 1]
        """
        probs = torch.sigmoid(logits)
        
        # Asymmetric probabilities
        p_pos = probs
        p_neg = 1.0 - probs
        
        # Asymmetric focusing
        if self.clip is not None and self.clip > 0:
            p_neg = torch.clamp(p_neg + self.clip, max=1.0)
            
        loss_pos = targets * torch.log(torch.clamp(p_pos, min=1e-7)) * ((1.0 - p_pos) ** self.gamma_pos)
        loss_neg = (1.0 - targets) * torch.log(torch.clamp(p_neg, min=1e-7)) * (probs ** self.gamma_neg)
        
        if self.pos_weight is not None:
            if not isinstance(self.pos_weight, torch.Tensor):
                self.pos_weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            loss_pos = loss_pos * self.pos_weight
            
        loss = - (loss_pos + loss_neg)
        return loss.mean()

class CalibratedSoftBCELoss(nn.Module):
    """
    Numerically stable Soft BCE for continuous consensus labels.
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        if self.pos_weight is not None:
            if not isinstance(self.pos_weight, torch.Tensor):
                self.pos_weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        return F.binary_cross_entropy_with_logits(logits, targets)

class JointSpaceOrdinalLoss(nn.Module):
    """
    Enforces that Medial and Lateral OA features maintain compartment specificity
    and penalizes severe inversion between joint compartments.
    """
    def __init__(self, base_loss_fn=None, margin_weight=0.1):
        super().__init__()
        self.base_loss_fn = base_loss_fn if base_loss_fn is not None else nn.BCEWithLogitsLoss()
        self.margin_weight = margin_weight

    def forward(self, logits, targets):
        base_loss = self.base_loss_fn(logits, targets)
        # Margin regularization: ensure individual head discriminability
        # If logits shape is (B, 3) for [MCL, Medial OA, Lateral OA]
        if logits.size(1) >= 3:
            medial_oa = logits[:, 1]
            lateral_oa = logits[:, 2]
            # When ground truth Medial OA >> Lateral OA, enforce margin
            oa_diff_target = targets[:, 1] - targets[:, 2]
            oa_diff_pred = medial_oa - lateral_oa
            # Hinge penalty if signs disagree strongly
            hinge = F.relu(0.5 - (oa_diff_target * oa_diff_pred))
            margin_loss = hinge.mean()
            return base_loss + self.margin_weight * margin_loss
        return base_loss
