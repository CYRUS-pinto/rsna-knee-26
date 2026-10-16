"""
Verification test and forward/backward gradient check for MultiScalePyramidalMIL.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import torch
from src.models.multiscale_pyramidal_transformer import MultiScalePyramidalMIL

def test_multiscale_transformer():
    print("=" * 80)
    print("TESTING MULTI-SCALE SPATIAL PYRAMIDAL TRANSFORMER (MS-SPT)")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiScalePyramidalMIL(num_classes=12, embed_dim=128, num_heads=4, dropout=0.1).to(device)
    
    # Synthetic inputs: B=2, K=4 slices
    local_x = torch.randn(2, 4, 3, 512, 512, device=device)
    global_x = torch.randn(2, 4, 3, 224, 224, device=device)
    
    print(f"Inputs: Local={local_x.shape} | Global={global_x.shape}")
    logits = model(local_x, global_x)
    
    assert logits.shape == (2, 12), f"Expected shape (2, 12), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Found NaNs in logits!"
    
    # Backward pass verification
    loss = logits.sum()
    loss.backward()
    
    print("Gradient backward pass: SUCCESS")
    print(f"Output logits shape: {logits.shape}")
    print("MULTI-SCALE PYRAMIDAL TRANSFORMER: VERIFIED!")
    print("=" * 80)

if __name__ == '__main__':
    test_multiscale_transformer()
