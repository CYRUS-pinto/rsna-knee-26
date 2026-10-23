import numpy as np

def apply_12x12_orthopaedic_coupling(scores: np.ndarray, include_acl_coupling: bool = False) -> np.ndarray:
    """
    Applies continuous, rank-preserving orthopaedic coupling.
    Eliminates triple rank-quantization while preserving exact continuous margins.
    """
    r = scores.copy()
    out = scores.copy()

    if include_acl_coupling:
        out[:, 10] = 0.970 * out[:, 10] + 0.030 * r[:, 0]
        out[:, 3]  = 0.980 * out[:, 3]  + 0.020 * r[:, 0]

    # 1. Effusion <-> Synovitis
    out[:, 7]  = 0.985 * out[:, 7]  + 0.015 * r[:, 8]
    out[:, 8]  = 0.980 * out[:, 8]  + 0.020 * r[:, 7]

    # 2. Effusion + Medial Meniscus -> Baker's
    out[:, 9]  = 0.975 * out[:, 9]  + 0.015 * r[:, 7]  + 0.010 * r[:, 2]

    # 3. Medial Meniscus <-> Medial OA
    out[:, 2]  = 0.985 * out[:, 2]  + 0.015 * r[:, 4]
    out[:, 4]  = 0.985 * out[:, 4]  + 0.015 * r[:, 2]

    # 4. Medial OA + Lateral OA -> PF OA
    out[:, 6]  = 0.980 * out[:, 6]  + 0.012 * r[:, 4]  + 0.008 * r[:, 5]

    # 5. Contusion + Effusion -> Fracture
    out[:, 11] = 0.975 * out[:, 11] + 0.015 * r[:, 10] + 0.010 * r[:, 7]

    return out
