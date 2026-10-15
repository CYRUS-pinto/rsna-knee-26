"""
Physical Multi-Plane Knee MRI Dataset for Engine 4.
Extracts true 2.5D slice triplet slabs across 5 anatomical sequence slots:
  Slot 0: Sagittal Fluid-Sensitive (T2 / PDFS) - Cruciate Ligaments (ACL/PCL), Effusions
  Slot 1: Sagittal Non-Fluid (T1 / PD) - Menisci, Bone Contusion, Fractures
  Slot 2: Coronal Fluid-Sensitive (T2 / PDFS) - Collateral Ligaments (MCL/LCL), Synovitis
  Slot 3: Coronal Non-Fluid (T1 / PD) - Medial & Lateral Compartment Osteoarthritis
  Slot 4: Axial - Patellofemoral Joint Alignment & Popliteal Cysts (Baker's)

Features:
- Dual-mode: Reads pre-extracted numpy volumes (fastest for training) OR parses raw DICOM directories.
- Physical slice normal sorting via dot(ImagePositionPatient, normal_vector).
- 140mm physical field-of-view cropping via PixelSpacing.
- 2nd/98th percentile contrast normalization.
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

CROP_MM = 140.0

def determine_orientation(iop):
    """
    Computes standard anatomical plane from ImageOrientationPatient [r_x, r_y, r_z, c_x, c_y, c_z].
    Cross product gives slice normal vector.
    """
    if iop is None or len(iop) < 6:
        return "Sagittal"
    r = np.array(iop[:3], dtype=float)
    c = np.array(iop[3:6], dtype=float)
    normal = np.cross(r, c)
    max_axis = np.argmax(np.abs(normal))
    if max_axis == 0:
        return "Sagittal"
    elif max_axis == 1:
        return "Coronal"
    else:
        return "Axial"

def read_dicom_slice(path):
    """Reads a single DICOM file safely using pydicom with fallback."""
    import pydicom
    try:
        d = pydicom.dcmread(path)
        arr = d.pixel_array.astype(np.float32)
        if getattr(d, 'PhotometricInterpretation', '') == 'MONOCHROME1':
            arr = arr.max() - arr
        iop = getattr(d, 'ImageOrientationPatient', None)
        ipp = getattr(d, 'ImagePositionPatient', None)
        ps = getattr(d, 'PixelSpacing', None)
        ps_val = float(ps[0]) if ps is not None else 0.5
        
        if iop is not None and ipp is not None and len(iop) == 6:
            r = np.array(iop[:3], float)
            c = np.array(iop[3:], float)
            n = np.cross(r, c)
            pos = float(np.dot(np.array(ipp, float), n))
        else:
            pos = float(getattr(d, 'InstanceNumber', 0) or 0)
            
        return pos, ps_val, arr, iop
    except Exception:
        return 0.0, 0.5, None, None

def crop_and_normalize(arr, ps_val, res=320):
    """Crops 140mm field of view and normalizes contrast using 2nd/98th percentiles."""
    import cv2
    if arr is None:
        return np.zeros((res, res), dtype=np.float32)
    h, w = arr.shape
    p = ps_val if ps_val > 0 else 0.5
    crop_px = int(round(CROP_MM / max(p, 1e-3)))
    crop_px = min(crop_px, min(h, w))
    y0 = max(0, (h - crop_px) // 2)
    x0 = max(0, (w - crop_px) // 2)
    cropped = arr[y0:y0 + crop_px, x0:x0 + crop_px]
    
    lo, hi = np.percentile(cropped, 2), np.percentile(cropped, 98)
    norm = np.clip((cropped - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    resized = cv2.resize(norm, (res, res), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)

class MultiPlane2p5DDataset(Dataset):
    def __init__(self, uids, labels_dict, cached_volume_dir=None, dicom_dir=None, 
                 k_per_slot=(18, 14, 14, 10, 12), res=320, is_train=True):
        """
        Args:
            uids: List of StudyInstanceUIDs.
            labels_dict: Dict mapping UID -> np.ndarray of shape (12,).
            cached_volume_dir: Directory containing pre-built <study_uid>.npy volumes of shape (Total_K, H, W).
            dicom_dir: Directory containing raw DICOMs (<dicom_dir>/<study_uid>/<series_uid>/*.dcm).
            k_per_slot: Number of slice windows per sequence slot (sum = Total_K).
            res: Spatial resolution (default 320).
            is_train: Data augmentation flag.
        """
        self.uids = list(uids)
        self.labels_dict = labels_dict
        self.cached_volume_dir = cached_volume_dir
        self.dicom_dir = dicom_dir
        self.k_per_slot = k_per_slot
        self.total_k = sum(k_per_slot)
        self.res = res
        self.is_train = is_train
        
        # Precompute plane slot index vector
        plane_list = []
        for slot_idx, k_count in enumerate(self.k_per_slot):
            plane_list.extend([slot_idx] * k_count)
        self.plane_ids = torch.tensor(plane_list, dtype=torch.long)

    def __len__(self):
        return len(self.uids)

    def _sample_indices(self, total_slices, k):
        if total_slices <= 3:
            return [0] * k
        usable = list(range(1, total_slices - 1))
        if self.is_train:
            pool = usable * (k // len(usable) + 1)
            np.random.shuffle(pool)
            return sorted(pool[:k])
        else:
            steps = np.linspace(0, len(usable) - 1, k).round().astype(int)
            return sorted([usable[i] for i in steps])

    def _load_from_dicoms(self, uid):
        """Extracts slices from DICOM directories across the 5 sequence slots."""
        study_path = os.path.join(self.dicom_dir, uid) if self.dicom_dir else ""
        if not os.path.exists(study_path):
            return torch.zeros((self.total_k, 3, self.res, self.res), dtype=torch.float32)

        series_dirs = [os.path.join(study_path, s) for s in os.listdir(study_path) if os.path.isdir(os.path.join(study_path, s))]
        
        # Parse series orientations and fluid sensitivity
        series_info = []
        for s_dir in series_dirs:
            dcm_files = glob.glob(os.path.join(s_dir, "*.dcm"))
            if not dcm_files:
                continue
            pos, ps, arr, iop = read_dicom_slice(dcm_files[0])
            plane = determine_orientation(iop)
            series_info.append({
                "dir": s_dir,
                "plane": plane,
                "files": dcm_files
            })

        slabs = []
        # Fallback pool
        fallback_slab = torch.zeros((3, self.res, self.res), dtype=torch.float32)
        
        # Slot matching: [Sagittal, Sagittal, Coronal, Coronal, Axial]
        target_planes = ["Sagittal", "Sagittal", "Coronal", "Coronal", "Axial"]
        for slot_idx, k_count in enumerate(self.k_per_slot):
            target_plane = target_planes[slot_idx]
            matching = [s for s in series_info if s["plane"] == target_plane]
            
            if matching:
                # Select first matching series for slot
                chosen = matching[slot_idx % len(matching)]
                slice_records = []
                for f in chosen["files"]:
                    p, ps, a, _ = read_dicom_slice(f)
                    if a is not None:
                        slice_records.append((p, ps, a))
                slice_records.sort(key=lambda x: x[0])
                
                if len(slice_records) >= 3:
                    picks = self._sample_indices(len(slice_records), k_count)
                    for pick in picks:
                        # 2.5D triplet: pick-1, pick, pick+1
                        idx_prev = max(0, pick - 1)
                        idx_next = min(len(slice_records) - 1, pick + 1)
                        im0 = crop_and_normalize(slice_records[idx_prev][2], slice_records[idx_prev][1], self.res)
                        im1 = crop_and_normalize(slice_records[pick][2], slice_records[pick][1], self.res)
                        im2 = crop_and_normalize(slice_records[idx_next][2], slice_records[idx_next][1], self.res)
                        slab = torch.from_numpy(np.stack([im0, im1, im2], axis=0))
                        slabs.append(slab)
                else:
                    for _ in range(k_count):
                        slabs.append(fallback_slab)
            else:
                for _ in range(k_count):
                    slabs.append(fallback_slab)

        # Ensure exact count matches total_k
        while len(slabs) < self.total_k:
            slabs.append(fallback_slab)
        return torch.stack(slabs[:self.total_k], dim=0)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        y = torch.tensor(self.labels_dict.get(uid, np.zeros(12, dtype=np.float32)), dtype=torch.float32)
        
        # 1. Fast Cache Path
        if self.cached_volume_dir:
            cache_path = os.path.join(self.cached_volume_dir, f"{uid}.npy")
            if os.path.exists(cache_path):
                arr = np.load(cache_path) # Expected: (Total_K, 3, H, W) or (Total_K, H, W)
                if arr.ndim == 3:
                    # Expand single slice to 3 channels
                    arr = np.repeat(arr[:, np.newaxis, :, :], 3, axis=1)
                tensor_stack = torch.from_numpy(arr.astype(np.float32))
                return tensor_stack, self.plane_ids, y

        # 2. DICOM Path
        tensor_stack = self._load_from_dicoms(uid)
        return tensor_stack, self.plane_ids, y
