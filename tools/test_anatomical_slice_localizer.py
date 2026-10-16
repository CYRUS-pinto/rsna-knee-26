"""
Unit test for deterministic anatomical slice localizer.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from src.models.anatomical_slice_localizer import locate_joint_center_slice, locate_fluid_slices

def test_localizer():
    print("=" * 80)
    print("TESTING ANATOMICAL SLICE LOCALIZER")
    print("=" * 80)
    
    # Synthetic 30-slice volume
    vol = np.random.randn(30, 256, 256).astype(np.float32)
    # Inject high-contrast edge in slice 15 (simulating joint line)
    vol[15, 120:136, 120:136] += 10.0
    
    center, slices = locate_joint_center_slice(vol)
    print(f"Detected Center Slice: {center} | Selected 12 Slices: {slices}")
    assert len(slices) == 12, f"Expected 12 slices, got {len(slices)}"
    assert 15 in slices, "Joint line slice 15 was not included in selection!"
    
    # Axial fluid test
    axial_vol = np.random.randn(25, 256, 256).astype(np.float32)
    axial_vol[10, 100:150, 100:150] += 50.0 # fluid pool
    fluid_slices = locate_fluid_slices(axial_vol)
    print(f"Selected Fluid Slices: {fluid_slices}")
    assert len(fluid_slices) == 12, f"Expected 12 slices, got {len(fluid_slices)}"
    assert 10 in fluid_slices, "Fluid pool slice 10 was not included in selection!"
    
    print("ANATOMICAL SLICE LOCALIZER: VERIFIED PASSED!")
    print("=" * 80)

if __name__ == '__main__':
    test_localizer()
