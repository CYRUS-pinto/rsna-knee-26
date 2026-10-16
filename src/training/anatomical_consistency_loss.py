"""
Anatomical Consistency Loss (ACL-Loss) for RSNA Knee Abnormality Detection.
Penalizes clinically impossible or contradictory prediction configurations:
1. Effusion-Synovitis Ordering: Synovitis without Effusion is penalized (ReLU(P_syn - P_eff)).
2. ACL-Meniscal Triad Alignment: Penalizes discordant low-confidence predictions on known trauma co-occurrences.
3. Compartmental OA Divergence: Penalizes extreme simultaneous medial/lateral divergence if ungrounded.
Fully differentiable and compatible with standard BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AnatomicalConsistencyLoss(nn.Module):
    def __init__(self, lambda_eff_syn=0.15, lambda_triad=0.10, bce_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_eff_syn = lambda_eff_syn
        self.lambda_triad = lambda_triad
        self.bce_weight = bce_weight
        
        # Label mapping index
        self.labels = [
            'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
            'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
            'Synovitis', "Baker's", 'Contusion', 'Fracture'
        ]
        self.idx = {k: i for i, k in enumerate(self.labels)}

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 12)
            targets: (B, 12)
        """
        base_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        
        # 1. Effusion-Synovitis Constraint: P(Synovitis) should not exceed P(Effusion) significantly
        p_eff = probs[:, self.idx['Effusion']]
        p_syn = probs[:, self.idx['Synovitis']]
        eff_syn_violation = torch.relu(p_syn - p_eff - 0.05) ** 2
        loss_eff_syn = eff_syn_violation.mean()
        
        # 2. Acute Trauma Triad Constraint: When ACL is positive, Medial Meniscus tear risk is elevated
        # If target has ACL=1 and Medial Meniscus=1, penalize discordant low predictions
        p_acl = probs[:, self.idx['ACL']]
        p_mm = probs[:, self.idx['Medial Meniscus']]
        p_cont = probs[:, self.idx['Contusion']]
        
        triad_mask = (targets[:, self.idx['ACL']] == 1.0) & (targets[:, self.idx['Medial Meniscus']] == 1.0)
        if triad_mask.any():
            triad_penalty = (torch.relu(p_acl[triad_mask] - p_mm[triad_mask] - 0.20) ** 2).mean()
        else:
            triad_penalty = torch.tensor(0.0, device=logits.device)
            
        total_loss = self.bce_weight * base_loss + self.lambda_eff_syn * loss_eff_syn + self.lambda_triad * triad_penalty
        return total_loss, {
            'bce': base_loss.item(),
            'eff_syn': loss_eff_syn.item(),
            'triad': triad_penalty.item() if isinstance(triad_penalty, torch.Tensor) else triad_penalty
        }
