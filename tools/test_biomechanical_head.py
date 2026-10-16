"""
Unit test and gradient flow verification for BiomechanicalRelationalHead.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import torch
from src.models.biomechanical_relational_head import BiomechanicalRelationalHead, ALL_LABELS

def test_biomechanical_head():
    print("=" * 80)
    print("TESTING BIOMECHANICAL RELATIONAL CROSS-ATTENTION HEAD")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BiomechanicalRelationalHead(num_classes=12, embed_dim=64, num_heads=4, dropout=0.1).to(device)
    
    # Synthetic batch of 8 studies
    logits = torch.randn(8, 12, requires_grad=True, device=device)
    
    out = model(logits)
    assert out.shape == (8, 12), f"Expected shape (8, 12), got {out.shape}"
    assert not torch.isnan(out).any(), "Found NaNs in output!"
    
    # Test gradient flow
    loss = out.sum()
    loss.backward()
    assert logits.grad is not None, "Logits did not receive gradients!"
    assert not torch.isnan(logits.grad).any(), "Found NaNs in gradients!"
    
    print(f"Forward shape: {out.shape} | Gradient norm: {logits.grad.norm().item():.4f}")
    print("Biomechanical Relational Head: VERIFIED PASSED!")
    print("=" * 80)

if __name__ == '__main__':
    test_biomechanical_head()
