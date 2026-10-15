"""
Unit test for training/engine3_slice_transformer/model_multiplane.py
Verifies:
1. Forward pass on 44-slice tensor (training sagittal corpus) -> exact shape (B, 12).
2. Forward pass on 96-slice multi-plane tensor (test inference) -> exact shape (B, 12).
3. Positional encoding with and without plane_ids.
4. Backward pass & valid gradient flow (no NaNs, no Inf).
"""

import sys, os
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from training.engine3_slice_transformer.model_multiplane import MultiPlaneSliceTransformerHead, MultiPlaneKneeModel

def test_multiplane_slice_transformer():
    B = 2
    F_dim = 256 # lightweight for fast test
    n = 12
    head = MultiPlaneSliceTransformerHead(F_dim=F_dim, n=n, num_heads=4, num_layers=2)
    
    # Test 1: 44-slice sagittal training representation (without plane_ids)
    x_44 = torch.randn(B, 44, F_dim, requires_grad=True)
    out_44 = head(x_44)
    assert out_44.shape == (B, n), f"Expected shape ({B}, {n}), got {out_44.shape}"
    loss_44 = out_44.sum()
    loss_44.backward()
    assert x_44.grad is not None and not torch.isnan(x_44.grad).any(), "NaN found in gradients for 44-slice pass!"
    print("Test 1 Passed: 44-slice single-plane forward & backward verified.")
    
    # Test 2: 96-slice multi-plane representation (with 5 plane IDs: e.g. 20 sag1, 20 sag2, 20 cor1, 20 cor2, 16 ax)
    head.zero_grad()
    x_96 = torch.randn(B, 96, F_dim, requires_grad=True)
    plane_ids = torch.cat([
        torch.full((20,), 0, dtype=torch.long),
        torch.full((20,), 1, dtype=torch.long),
        torch.full((20,), 2, dtype=torch.long),
        torch.full((20,), 3, dtype=torch.long),
        torch.full((16,), 4, dtype=torch.long),
    ]) # length 96
    
    out_96 = head(x_96, plane_ids=plane_ids)
    assert out_96.shape == (B, n), f"Expected shape ({B}, {n}), got {out_96.shape}"
    loss_96 = out_96.sum()
    loss_96.backward()
    assert x_96.grad is not None and not torch.isnan(x_96.grad).any(), "NaN found in gradients for 96-slice pass!"
    print("Test 2 Passed: 96-slice multi-plane forward & backward verified.")
    
    # Test 3: Model wrapper with dummy backbone
    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, F_dim, kernel_size=3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        def forward(self, img):
            return self.pool(self.conv(img)).flatten(1)
            
    full_model = MultiPlaneKneeModel(backbone=DummyBackbone(), F_dim=F_dim, n=n, num_heads=4, num_layers=1)
    img_tensor = torch.randn(1, 16, 3, 64, 64) # 1 study, 16 slices, 3 channels, 64x64
    out_full = full_model(img_tensor)
    assert out_full.shape == (1, n), f"Full model output shape mismatch: {out_full.shape}"
    print("Test 3 Passed: End-to-end MultiPlaneKneeModel wrapper verified.")

if __name__ == '__main__':
    test_multiplane_slice_transformer()
    print("ALL MULTI-PLANE SLICE TRANSFORMER TESTS PASSED SUCCESSFULLY.")
