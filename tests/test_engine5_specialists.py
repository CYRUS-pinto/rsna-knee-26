"""
Unit tests for Engine 5 Decoupled Specialists and Loss Functions (ADR-015).
"""

import pytest
import torch
import numpy as np
from src.models.decoupled_specialists import SagittalSpecialistMIL, CoronalSpecialistMIL, AxialFluidSpecialistMIL
from src.models.losses import AsymmetricFocalLoss, JointSpaceOrdinalLoss, CalibratedSoftBCELoss
from src.data.plane_datasets import PlaneSpecialistDataset, SPECIALIST_TARGETS

def test_sagittal_specialist_forward():
    model = SagittalSpecialistMIL(backbone_name='resnet18', pretrained=False)
    x = torch.randn(2, 8, 3, 64, 64) # (B=2, K=8, C=3, H=64, W=64)
    out = model(x)
    assert out.shape == (2, 4), f"Expected shape (2, 4), got {out.shape}"
    assert not torch.isnan(out).any()

def test_coronal_specialist_forward():
    model = CoronalSpecialistMIL(backbone_name='resnet18', pretrained=False)
    x = torch.randn(2, 6, 3, 64, 64) # (B=2, K=6, C=3, H=64, W=64)
    out = model(x)
    assert out.shape == (2, 3), f"Expected shape (2, 3), got {out.shape}"
    assert not torch.isnan(out).any()

def test_axial_specialist_forward():
    model = AxialFluidSpecialistMIL(backbone_name='resnet18', pretrained=False)
    x = torch.randn(2, 4, 3, 64, 64) # (B=2, K=4, C=3, H=64, W=64)
    out = model(x)
    assert out.shape == (2, 5), f"Expected shape (2, 5), got {out.shape}"
    assert not torch.isnan(out).any()

def test_asymmetric_focal_loss():
    loss_fn = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=2.0)
    logits = torch.randn(4, 4, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.5, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [1.0, 1.0, 0.0, 1.0]])
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None

def test_joint_space_ordinal_loss():
    loss_fn = JointSpaceOrdinalLoss(margin_weight=0.1)
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([[0.0, 1.0, 0.0],
                            [0.5, 0.0, 0.8],
                            [0.0, 0.9, 0.1],
                            [0.0, 0.0, 0.0]])
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None

def test_calibrated_soft_bce_loss():
    loss_fn = CalibratedSoftBCELoss()
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.rand(4, 5)
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None

def test_plane_specialist_dataset():
    uids = ["1.2.3.4", "1.2.3.5"]
    labels_dict = {
        "1.2.3.4": np.ones(12, dtype=np.float32),
        "1.2.3.5": np.zeros(12, dtype=np.float32)
    }
    ds_sag = PlaneSpecialistDataset(uids, labels_dict, plane='sagittal', is_train=False)
    assert len(ds_sag) == 2
    x, y = ds_sag[0]
    assert x.shape == (32, 3, 320, 320)
    assert y.shape == (4,)
    assert (y == 1.0).all()

    ds_cor = PlaneSpecialistDataset(uids, labels_dict, plane='coronal', is_train=False)
    x, y = ds_cor[1]
    assert x.shape == (24, 3, 320, 320)
    assert y.shape == (3,)
    assert (y == 0.0).all()

    ds_ax = PlaneSpecialistDataset(uids, labels_dict, plane='axial', is_train=False)
    x, y = ds_ax[0]
    assert x.shape == (12, 3, 320, 320)
    assert y.shape == (5,)
