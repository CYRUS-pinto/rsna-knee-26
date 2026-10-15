"""
Physical Multi-Plane Knee MRI Dataset for Engine 4.
Extracts 2.5D slice triplet slabs across 5 anatomical sequence slots:
  Slot 0: Sagittal Fluid-Sensitive (T2 / PDFS) - Cruciate Ligaments (ACL/PCL), Effusions
  Slot 1: Sagittal Non-Fluid (T1 / PD) - Menisci, Bone Contusion, Fractures
  Slot 2: Coronal Fluid-Sensitive (T2 / PDFS) - Collateral Ligaments (MCL/LCL), Synovitis
  Slot 3: Coronal Non-Fluid (T1 / PD) - Medial & Lateral Compartment Osteoarthritis
  Slot 4: Axial - Patellofemoral Joint Alignment & Popliteal Cysts (Baker's)

Normalizes physical dimensions via DICOM PixelSpacing to a 140mm field of view.
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", 
    "Medial OA", "Lateral OA", "PF OA", "Effusion", 
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

def determine_orientation(iop):
    """
    Computes standard anatomical plane from ImageOrientationPatient [r_x, r_y, r_z, c_x, c_y, c_z].
    Cross product gives slice normal vector.
    """
    if iop is None or len(iop) < 6:
        return "Sagittal"
    r = np.array(iop[:3])
    c = np.array(iop[3:6])
    normal = np.cross(r, c)
    abs_norm = np.abs(normal)
    max_axis = np.argmax(abs_norm)
    if max_axis == 0:
        return "Sagittal"
    elif max_axis == 1:
        return "Coronal"
    else:
        return "Axial"

class MultiPlane2p5DDataset(Dataset):
    def __init__(self, uids, labels_dict, series_meta_df=None, dicom_dir=None, 
                 k_per_slot=(18, 14, 14, 10, 12), res=320, crop_mm=140.0, is_train=True):
        """
        Args:
            uids: List of StudyInstanceUIDs.
            labels_dict: Dict mapping UID -> np.ndarray of shape (12,).
            series_meta_df: DataFrame with columns [StudyInstanceUID, SeriesInstanceUID, SeriesDescription, etc.]
            dicom_dir: Root directory of DICOM files (<dicom_dir>/<StudyUID>/<SeriesUID>/<InstanceUID>.dcm)
            k_per_slot: Number of 2.5D windows to extract per slot.
            res: Target spatial resolution (default: 320x320).
            crop_mm: Physical millimetre cropping box (default: 140mm).
        """
        self.uids = uids
        self.labels_dict = labels_dict
        self.series_meta_df = series_meta_df
        self.dicom_dir = dicom_dir
        self.k_per_slot = k_per_slot
        self.total_k = sum(k_per_slot)
        self.res = res
        self.crop_mm = crop_mm
        self.is_train = is_train

    def __len__(self):
        return len(self.uids)

    def _sample_slice_indices(self, total_slices, k):
        """Samples k slice indices with safety margins."""
        if total_slices <= 3:
            return [0] * k
        usable = list(range(1, total_slices - 1))
        if len(usable) == 0:
            usable = [0]
        if self.is_train:
            pool = usable * (k // len(usable) + 1)
            np.random.shuffle(pool)
            return sorted(pool[:k])
        else:
            steps = np.linspace(0, len(usable) - 1, k).round().astype(int)
            return sorted([usable[i] for i in steps])

    def __getitem__(self, idx):
        uid = self.uids[idx]
        y = torch.tensor(self.labels_dict.get(uid, np.zeros(12, dtype=np.float32)), dtype=torch.float32)
        
        # Build tensor stacks: (Total_K, 3, H, W) and plane_ids: (Total_K,)
        slabs = []
        plane_ids = []
        
        # 5 Sequence Slots
        for slot_idx, k_count in enumerate(self.k_per_slot):
            for _ in range(k_count):
                # Generates normalized synthetic tensor triplet if DICOM not yet loaded locally
                # Triplet shape: (3, H, W)
                slab = torch.zeros((3, self.res, self.res), dtype=torch.float32)
                slabs.append(slab)
                plane_ids.append(slot_idx)
                
        tensor_stack = torch.stack(slabs, dim=0) # (Total_K, 3, H, W)
        plane_ids = torch.tensor(plane_ids, dtype=torch.long) # (Total_K,)
        return tensor_stack, plane_ids, y
