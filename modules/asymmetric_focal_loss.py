"""
Asymmetric Focal Loss (ASL) for Multi-Label Medical Imaging
Reference: Ridnik et al. (ICCV 2021) "Asymmetric Loss For Multi-Label Classification"
Optimized for extreme class imbalance in knee MRI findings (Fractures, Contusions).
"""

import torch
import torch.nn as nn

class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Loss that decouples positive and negative focusing parameters.
    Suppresses easy negative gradients (gamma_neg = 4.0) while preserving positive gradients (gamma_pos = 1.0 or 0.0).
    Includes probability margin clipping to completely discard very easy negative samples.
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw unnormalized logits
            targets: (B, C) binary labels [0, 1]
        """
        # Calculating Probabilities
        p = torch.sigmoid(logits)
        p_pos = p
        p_neg = 1.0 - p

        # Asymmetric Clipping for Negatives
        if self.clip is not None and self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)

        # Basic Cross Entropy Calculation
        los_pos = targets * torch.log(p_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(p_neg.clamp(min=self.eps))

        # Asymmetric Focusing Weights
        if self.gamma_pos > 0:
            los_pos = los_pos * ((1.0 - p_pos) ** self.gamma_pos)
        if self.gamma_neg > 0:
            los_neg = los_neg * (p ** self.gamma_neg)

        loss = -los_pos - los_neg
        return loss.mean()

if __name__ == '__main__':
    # Unit test
    criterion = AsymmetricFocalLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    logits = torch.randn(4, 12, requires_grad=True)
    targets = torch.randint(0, 2, (4, 12)).float()
    loss = criterion(logits, targets)
    loss.backward()
    assert torch.isfinite(loss), "Loss must be finite"
    assert logits.grad is not None, "Gradients must exist"
    print(f"[+] AsymmetricFocalLoss unit test passed! Loss: {loss.item():.4f}")
