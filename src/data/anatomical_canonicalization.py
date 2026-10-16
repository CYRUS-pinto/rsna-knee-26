"""
Anatomical Laterality Canonicalization & Mirror Augmentation Pipeline.
Implements Pillar 2 of ADR-013:
1. Patient-Coordinate Centroid Laterality Detection (resolves missing Laterality tag for >97% of studies).
2. Canonicalizes Coronal & Axial slices so Medial is ALWAYS on the left, Lateral is ALWAYS on the right.
3. Anatomical Mirror Augmentation: When horizontally flipped during training, swaps Medial/Lateral targets,
   effectively doubling training data for rare classes (e.g. Lateral OA) while preserving anatomical truth.
"""

import numpy as np
import torch

# Compartmental findings that swap under horizontal reflection
SWAP_PAIRS = [
    ('Medial Meniscus', 'Lateral Meniscus'),
    ('Medial OA', 'Lateral OA')
]

ALL_LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def resolve_laterality(tag_laterality=None, ipp_list=None):
    """
    Resolves knee laterality ('R' or 'L') with fallback to ImagePositionPatient X-coordinate.
    DICOM standard: Patient coordinate X axis goes from Right (-X) to Left (+X).
    A knee centered at x < 0 is Right; x > 0 is Left.
    """
    if tag_laterality and str(tag_laterality).strip().upper() in ('R', 'RIGHT'):
        return 'R'
    if tag_laterality and str(tag_laterality).strip().upper() in ('L', 'LEFT'):
        return 'L'
        
    if ipp_list is not None and len(ipp_list) > 0:
        # Compute mean X position across series slices
        x_coords = [float(pos[0]) for pos in ipp_list if pos is not None and len(pos) >= 1]
        if x_coords:
            mean_x = float(np.mean(x_coords))
            return 'L' if mean_x > 0 else 'R'
            
    return 'R' # Default fallback

def canonicalize_slice_orientation(image, laterality, plane='Coronal'):
    """
    Horizontally flips slices so Medial compartment is always positioned on the left side.
    In standard Coronal acquisition of a Left Knee, medial is on the right of the pixel matrix.
    Flipping horizontally standardizes the medial compartment to the left.
    """
    if image is None:
        return None
    if plane.lower() in ('coronal', 'axial') and laterality == 'L':
        # Flip width axis (last axis)
        if isinstance(image, np.ndarray):
            return np.flip(image, axis=-1).copy()
        elif torch.is_tensor(image):
            return torch.flip(image, dims=[-1])
    return image

def apply_anatomical_mirror_augmentation(image_tensor, target_dict, p=0.5):
    """
    Applies anatomical horizontal flip augmentation during training.
    When flipped, compartmental targets are swapped to preserve physiological truth:
      - Medial Meniscus <-> Lateral Meniscus
      - Medial OA <-> Lateral OA
    """
    if np.random.rand() > p:
        return image_tensor, target_dict, False
        
    # Flip horizontal axis
    if torch.is_tensor(image_tensor):
        flipped_img = torch.flip(image_tensor, dims=[-1])
    else:
        flipped_img = np.flip(image_tensor, axis=-1).copy()
        
    # Swap compartmental targets
    flipped_targets = dict(target_dict)
    for c1, c2 in SWAP_PAIRS:
        if c1 in flipped_targets and c2 in flipped_targets:
            flipped_targets[c1], flipped_targets[c2] = target_dict[c2], target_dict[c1]
            
    return flipped_img, flipped_targets, True
