import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

def test_dinov2_encoder_dimensions_and_masking():
    import torch
    from src.models.dinov2_25d_encoder import DINOv2KneeClassifier, TARGETS
    
    # Batch = 2 studies, K = 8 slice packets, Channels = 3, Resolution = 224x224
    B, K, C, H, W = 2, 8, 3, 224, 224
    dummy_x = torch.randn(B, K, C, H, W)
    dummy_mask = torch.ones(B, K)
    # Mask out the last 2 slices for study 1
    dummy_mask[1, 6:] = 0.0
    
    model = DINOv2KneeClassifier(embed_dim=384, num_targets=len(TARGETS))
    model.eval()
    
    with torch.no_grad():
        logits = model(dummy_x, mask=dummy_mask)
        
    assert logits.shape == (B, 12), f"Expected shape ({B}, 12), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaNs!"
    assert not torch.isinf(logits).any(), "Logits contain Infs!"
    
    print(f"PASS: DINOv2KneeClassifier forward pass verified. Logits shape: {logits.shape}")

if __name__ == "__main__":
    test_dinov2_encoder_dimensions_and_masking()
