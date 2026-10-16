import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath('.'))
from src.data.anatomical_canonicalization import (
    resolve_laterality,
    canonicalize_slice_orientation,
    apply_anatomical_mirror_augmentation,
    SWAP_PAIRS
)

def test_laterality_and_mirroring():
    print("Testing resolve_laterality...")
    assert resolve_laterality(tag_laterality='R') == 'R'
    assert resolve_laterality(tag_laterality='L') == 'L'
    # Fallback to X coordinate
    assert resolve_laterality(ipp_list=[[-120.5, 50.0, 10.0]]) == 'R'
    assert resolve_laterality(ipp_list=[[115.0, 50.0, 10.0]]) == 'L'
    print("Laterality resolution passed!")

    print("Testing canonicalize_slice_orientation...")
    dummy_slice = np.array([[1, 2], [3, 4]], dtype=np.float32)
    # Right knee unchanged
    r_slice = canonicalize_slice_orientation(dummy_slice, laterality='R', plane='Coronal')
    assert np.array_equal(r_slice, dummy_slice)
    # Left knee flipped horizontally
    l_slice = canonicalize_slice_orientation(dummy_slice, laterality='L', plane='Coronal')
    expected_flip = np.array([[2, 1], [4, 3]], dtype=np.float32)
    assert np.array_equal(l_slice, expected_flip)
    print("Orientation canonicalization passed!")

    print("Testing apply_anatomical_mirror_augmentation...")
    targets = {
        'ACL': 0.8,
        'Medial Meniscus': 0.9,
        'Lateral Meniscus': 0.1,
        'Medial OA': 0.7,
        'Lateral OA': 0.2,
        'Effusion': 0.95
    }
    img_tensor = torch.randn(3, 64, 64)
    # Force flip with p=1.0
    flipped_img, flipped_targets, is_flipped = apply_anatomical_mirror_augmentation(img_tensor, targets, p=1.0)
    assert is_flipped is True
    # Verify swapped targets
    assert flipped_targets['Medial Meniscus'] == 0.1
    assert flipped_targets['Lateral Meniscus'] == 0.9
    assert flipped_targets['Medial OA'] == 0.2
    assert flipped_targets['Lateral OA'] == 0.7
    # Verify non-compartmental targets stay unchanged
    assert flipped_targets['ACL'] == 0.8
    assert flipped_targets['Effusion'] == 0.95
    print("Anatomical mirror augmentation passed!")

if __name__ == '__main__':
    test_laterality_and_mirroring()
    print("ALL TESTS PASSED CLEANLY!")
