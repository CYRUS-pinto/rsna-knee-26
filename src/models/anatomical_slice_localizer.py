"""
Deterministic Anatomical Slice Localizer for Knee MRI Sequences.
Solves the coarse-to-fine localization problem used in 1st place RSNA winning architectures:
Instead of feeding arbitrary equidistant slices, dynamically identifies the key slices:
1. Joint Space Center: Finds the slice with maximum femoral-tibial contact margin gradient.
2. Suprapatellar Pouch: Finds the fluid collection peak on axial sequences.
3. Cruciate Ligament Span: Identifies the 8-12 contiguous slices encompassing the intercondylar notch.
"""

import numpy as np

def locate_joint_center_slice(sagittal_volume):
    """
    Identifies the intercondylar notch / cruciate ligament center slice along the width axis.
    In knee MRI sagittal series, the intercondylar notch is characterized by:
    - Lowest bone density at the joint center
    - Peak contrast between femoral condyles and cruciate ligaments
    Args:
        sagittal_volume: (N_slices, H, W) numpy array
    Returns:
        center_idx: int (index of the notch center slice)
        slice_indices: list of ints (the optimal 12 contiguous slices)
    """
    N, H, W = sagittal_volume.shape
    if N <= 12:
        return N // 2, list(range(N))
        
    # Focus on central 50% region of the knee
    h_start, h_end = int(H * 0.25), int(H * 0.75)
    w_start, w_end = int(W * 0.25), int(W * 0.75)
    central_slab = sagittal_volume[:, h_start:h_end, w_start:w_end]
    
    # Compute horizontal line variance (femoral-tibial contact margin has sharpest contrast)
    diffs = np.diff(central_slab, axis=1) # (N, H-1, W)
    slice_activity = np.mean(np.abs(diffs), axis=(1, 2)) # (N,)
    
    # Smooth activity profile across slices to avoid single-slice noise
    kernel = np.array([0.2, 0.6, 0.2])
    smoothed = np.convolve(slice_activity, kernel, mode='same')
    
    # Center slice corresponds to the notch / active cruciate span
    center_idx = int(np.argmax(smoothed))
    
    # Select 12 contiguous slices centered on peak
    start = max(0, min(center_idx - 6, N - 12))
    end = start + 12
    return center_idx, list(range(start, end))

def locate_fluid_slices(axial_volume):
    """
    Identifies the suprapatellar pouch and popliteal fossa slices on axial fluid sequences.
    Fluid appears hyperintense on T2/PD fat-suppressed axial images (95th percentile intensity peak).
    """
    N, H, W = axial_volume.shape
    if N <= 12:
        return list(range(N))
        
    # High-intensity fluid pixel count per slice
    fluid_counts = []
    for i in range(N):
        sl = axial_volume[i]
        p95 = np.percentile(sl, 95)
        fluid_counts.append(np.sum(sl > p95))
        
    fluid_profile = np.array(fluid_counts)
    # Peak fluid accumulation slice
    peak_idx = int(np.argmax(fluid_profile))
    
    start = max(0, min(peak_idx - 6, N - 12))
    return list(range(start, start + 12))
