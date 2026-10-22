"""
Physical Millimeter-Space Knee MRI Dataset Loader & Normalizer.
Implements:
1. Physical Window Cropping: 130 mm invariant anatomical field of view based on DICOM PixelSpacing.
2. 2.5D Triplets: Consecutive adjacent slices [c-1, c, c+1] for capturing volumetric through-plane context.
3. Central Stack Sampling: Extracts from the central 20% to 80% of the joint space to avoid empty slices.
4. Robust Percentile Contrast Normalization: 1st/99th percentile windowing invariant to scanner vendor.
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

TARGET_FOV_MM = 130.0  # 130 mm fixed physical window covering typical adult knee joint margin
DEFAULT_PIXEL_SPACING = 0.40  # mm/pixel default

def compute_physical_crop_box(arr, pixel_spacing_mm=None, target_size=224, target_fov_mm=TARGET_FOV_MM):
    """
    Computes a centered crop window whose physical footprint is invariant across scan resolutions.
    """
    if arr is None:
        return np.zeros((target_size, target_size), dtype=np.float32)
        
    H, W = arr.shape
    if pixel_spacing_mm is None or pixel_spacing_mm <= 0:
        pixel_spacing_mm = DEFAULT_PIXEL_SPACING
        
    # Desired crop width/height in pixels
    desired_pixels = int(round(target_fov_mm / pixel_spacing_mm))
    desired_pixels = max(32, min(desired_pixels, min(H, W)))
    
    # Joint center estimation via vertical/horizontal intensity center-of-mass
    # Restricted to the central 25%-75% of the frame
    y_profile = arr.sum(axis=1)
    x_profile = arr.sum(axis=0)
    
    y_min, y_max = int(H * 0.25), int(H * 0.75)
    x_min, x_max = int(W * 0.25), int(W * 0.75)
    
    cy = y_min + int(np.argmax(y_profile[y_min:y_max])) if y_max > y_min else H // 2
    cx = x_min + int(np.argmax(x_profile[x_min:x_max])) if x_max > x_min else W // 2
    
    half_box = desired_pixels // 2
    y0 = max(0, min(cy - half_box, H - desired_pixels))
    x0 = max(0, min(cx - half_box, W - desired_pixels))
    
    crop = arr[y0:y0 + desired_pixels, x0:x0 + desired_pixels]
    
    # 1st/99th percentile normalization
    p1, p99 = np.percentile(crop, 1.0), np.percentile(crop, 99.0)
    norm = np.clip((crop - p1) / max(p99 - p1, 1e-5), 0.0, 1.0)
    
    # Resize to target input resolution (e.g. 224x224)
    im = Image.fromarray((norm * 255).astype(np.uint8)).resize((target_size, target_size), Image.BILINEAR)
    return np.array(im, dtype=np.float32) / 127.5 - 1.0

def assemble_25d_packet(slice_stack, center_idx):
    """
    Assembles a 3-channel 2.5D triplet [c-1, c, c+1].
    """
    N = len(slice_stack)
    if N == 0:
        return None
    idx_prev = max(0, center_idx - 1)
    idx_curr = max(0, min(center_idx, N - 1))
    idx_next = min(N - 1, center_idx + 1)
    
    return np.stack([
        slice_stack[idx_prev],
        slice_stack[idx_curr],
        slice_stack[idx_next]
    ], axis=0)  # Shape: (3, H, W)
