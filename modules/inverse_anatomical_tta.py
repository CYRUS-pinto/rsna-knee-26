"""
Inverse Anatomical Test-Time Augmentation (TTA) Module.
Crucial Inductive Bias for Knee MRI:
When a knee image is horizontally flipped (x -> -x), the anatomical medial and lateral sides invert:
- Medial Meniscus becomes Lateral Meniscus.
- Lateral Meniscus becomes Medial Meniscus.
- Medial OA becomes Lateral OA.
- Lateral OA becomes Medial OA.
Standard naive TTA averages P(X) and P(Flip(X)) identically across all columns,
which causes cross-compartment contamination!
This module applies Anatomical Inversion before TTA averaging.
"""

import numpy as np
import pandas as pd
import torch

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# Mapping of columns to swap when horizontally flipped
SWAP_MAP = {
    'Medial Meniscus': 'Lateral Meniscus',
    'Lateral Meniscus': 'Medial Meniscus',
    'Medial OA': 'Lateral OA',
    'Lateral OA': 'Medial OA'
}

def apply_anatomical_flip_tta(p_orig: np.ndarray, p_flipped: np.ndarray, targets=TARGETS, weight_orig: float = 0.5) -> np.ndarray:
    """
    Combines original predictions and flipped predictions with anatomical compartment inversion.
    
    Args:
        p_orig: Array of shape (N, C) containing predictions for original images.
        p_flipped: Array of shape (N, C) containing predictions for horizontally flipped images.
        targets: List of target column names.
        weight_orig: Weight for original prediction (default 0.5).
        
    Returns:
        p_tta: Corrected TTA prediction array of shape (N, C).
    """
    assert p_orig.shape == p_flipped.shape, "Shape mismatch between orig and flipped!"
    p_tta = np.zeros_like(p_orig)
    
    col_idx = {col: i for i, col in enumerate(targets)}
    
    for i, col in enumerate(targets):
        if col in SWAP_MAP:
            # Swap with the opposing compartment
            partner_col = SWAP_MAP[col]
            partner_idx = col_idx[partner_col]
            # p_flipped on partner_col corresponds to col in real anatomy!
            flipped_corrected = p_flipped[:, partner_idx]
        else:
            # Symmetric or central findings (ACL, MCL, Effusion, Synovitis, Baker's, Fracture)
            flipped_corrected = p_flipped[:, i]
            
        p_tta[:, i] = weight_orig * p_orig[:, i] + (1.0 - weight_orig) * flipped_corrected
        
    return p_tta

if __name__ == '__main__':
    # Unit test
    N, C = 5, 12
    p_orig = np.random.uniform(0.1, 0.9, (N, C))
    p_flip = np.random.uniform(0.1, 0.9, (N, C))
    
    p_tta = apply_anatomical_flip_tta(p_orig, p_flip)
    assert p_tta.shape == (N, C), f"Expected {(N, C)}, got {p_tta.shape}"
    
    # Check that Medial Meniscus used Lateral Meniscus from p_flip
    m_idx = TARGETS.index('Medial Meniscus')
    l_idx = TARGETS.index('Lateral Meniscus')
    expected_m = 0.5 * p_orig[:, m_idx] + 0.5 * p_flip[:, l_idx]
    np.testing.assert_allclose(p_tta[:, m_idx], expected_m)
    print("[+] Inverse Anatomical TTA unit test passed perfectly!")
