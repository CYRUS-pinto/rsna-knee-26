import os
import pydicom
import numpy as np
from typing import Dict, List, Tuple

def parse_study_series_metadata(study_dir: str) -> Dict[str, Tuple[str, int, int]]:
    """Fast header scan extracting plane, fluid-sensitivity, and slice count per series."""
    meta = {}
    if not os.path.isdir(study_dir):
        return meta
    for s_name in os.listdir(study_dir):
        s_path = os.path.join(study_dir, s_name)
        if not os.path.isdir(s_path):
            continue
        dcms = [f for f in os.listdir(s_path) if f.endswith(".dcm")]
        if not dcms:
            continue
        # Read middle slice header
        mid_file = os.path.join(s_path, dcms[len(dcms) // 2])
        try:
            ds = pydicom.dcmread(mid_file, stop_before_pixels=True, force=True)
            iop = getattr(ds, "ImageOrientationPatient", None)
            plane = "Sagittal"
            if iop is not None and len(iop) == 6:
                norm = np.cross(np.array(iop[:3]), np.array(iop[3:6]))
                dom = int(np.argmax(np.abs(norm)))
                plane = "Sagittal" if dom == 0 else ("Coronal" if dom == 1 else "Axial")
            te = float(getattr(ds, "EchoTime", 0.0) or 0.0)
            fs = 1 if te >= 28.0 else 0
            meta[s_name] = (plane, fs, len(dcms))
        except Exception:
            meta[s_name] = ("Sagittal", 0, len(dcms))
    return meta
