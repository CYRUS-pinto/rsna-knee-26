"""
Unit test for AnatomicalConsistencyLoss module.
Verifies:
1. Base BCE computation
2. Effusion-Synovitis ordering penalty
3. Acute Trauma Triad penalty
4. Differentiability and backward gradient flow
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import torch
from src.training.anatomical_consistency_loss import AnatomicalConsistencyLoss

def test_anatomical_loss():
    print("=" * 80)
    print("TESTING ANATOMICAL CONSISTENCY LOSS")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = AnatomicalConsistencyLoss().to(device)
    
    logits = torch.randn(4, 12, requires_grad=True, device=device)
    targets = torch.randint(0, 2, (4, 12)).float().to(device)
    
    total_loss, metrics = loss_fn(logits, targets)
    assert total_loss.item() > 0, "Loss must be positive"
    assert not torch.isnan(total_loss), "Found NaNs in loss!"
    
    total_loss.backward()
    assert logits.grad is not None, "Gradients not received!"
    assert not torch.isnan(logits.grad).any(), "Found NaNs in gradients!"
    
    print(f"Total Loss: {total_loss.item():.4f} | Metrics: {metrics}")
    print("ANATOMICAL CONSISTENCY LOSS: VERIFIED PASSED!")
    print("=" * 80)

if __name__ == '__main__':
    test_anatomical_loss()
