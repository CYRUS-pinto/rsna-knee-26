import torch
import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from src.models.sequence_mil_3d import VolumetricSequenceMIL

def test_sequence_mil_shape_and_grad():
    print("Testing VolumetricSequenceMIL forward pass and gradient backpropagation...")
    # Using tiny backbone with pretrained=False for fast local testing
    model = VolumetricSequenceMIL(
        num_classes=4,
        backbone_name='resnet18',
        pretrained=False,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.1
    )
    model.train()
    
    # Batch of 2 studies, 8 sequential slices, 3 channels (c-1, c, c+1), 128x128
    dummy_input = torch.randn(2, 8, 3, 128, 128, requires_grad=True)
    out = model(dummy_input)
    
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 4), f"Expected shape (2, 4), got {out.shape}"
    
    loss = out.sum()
    loss.backward()
    assert dummy_input.grad is not None, "Gradient was not propagated back to input!"
    print("Test passed successfully! Output shape and backprop verified.")

if __name__ == '__main__':
    test_sequence_mil_shape_and_grad()
