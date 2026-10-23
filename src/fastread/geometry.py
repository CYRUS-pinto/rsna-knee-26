import numpy as np

def compute_slice_normal(iop: list) -> np.ndarray:
    """Computes slice normal vector via row and column direction cosine cross product."""
    row = np.array(iop[:3], dtype=np.float64)
    col = np.array(iop[3:6], dtype=np.float64)
    norm = np.cross(row, col)
    return norm / np.linalg.norm(norm)

def project_slice_depth(ipp: list, normal: np.ndarray) -> float:
    """Projects ImagePositionPatient onto normal vector to yield monotonic slice depth."""
    return float(np.dot(np.array(ipp, dtype=np.float64), normal))

def compute_metric_crop_box(pixel_spacing: list, fov_mm: float = 140.0) -> tuple:
    """Computes bounding box pixel dimensions for fixed physical FOV."""
    sy, sx = float(pixel_spacing[0]), float(pixel_spacing[1])
    ch = int(round(fov_mm / max(sy, 1e-3)))
    cw = int(round(fov_mm / max(sx, 1e-3)))
    return ch, cw
