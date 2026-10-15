"""
Automated Unit Tests for Engine 4 Data & Model Pipeline.
"""

import pytest
import torch
import numpy as np
from src.data.multiplane_dataset import MultiPlane2p5DDataset, LABELS
from src.models.engine4_convnext_mil import Engine4ConvNeXtMIL, PerFindingAttentionMILHead

def test_dataset_output_shapes():
    uids = ["study_001", "study_002"]
    labels_dict = {
        "study_001": np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
        "study_002": np.zeros(12, dtype=np.float32)
    }
    k_per_slot = (4, 4, 4, 2, 2) # Total 16 windows for fast unit test
    ds = MultiPlane2p5DDataset(uids, labels_dict, k_per_slot=k_per_slot, res=128)
    
    assert len(ds) == 2
    x, plane_ids, y = ds[0]
    
    assert x.shape == (16, 3, 128, 128)
    assert plane_ids.shape == (16,)
    assert y.shape == (12,)
    assert (plane_ids >= 0).all() and (plane_ids <= 4).all()

def test_head_forward_and_backward():
    B, K, D = 2, 16, 64
    head = PerFindingAttentionMILHead(in_features=D, num_classes=12, hidden_dim=32, num_planes=5)
    
    dummy_feats = torch.randn(B, K, D, requires_grad=True)
    dummy_planes = torch.randint(0, 5, (B, K))
    
    logits = head(dummy_feats, plane_ids=dummy_planes)
    assert logits.shape == (B, 12)
    
    loss = logits.sum()
    loss.backward()
    assert dummy_feats.grad is not None

def test_engine4_model_forward():
    # Use small dummy backbone config or direct mock for speed
    B, K, C, H, W = 1, 8, 3, 64, 64
    model = Engine4ConvNeXtMIL(backbone_name='resnet10t', pretrained=False, num_classes=12, hidden_dim=32)
    
    x = torch.randn(B, K, C, H, W)
    plane_ids = torch.randint(0, 5, (K,))
    
    logits = model(x, plane_ids=plane_ids)
    assert logits.shape == (B, 12)
    assert torch.isfinite(logits).all()
