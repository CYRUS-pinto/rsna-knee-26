import os
import sys
import numpy as np
import pandas as pd

ALL_LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

SPECIALIST_TARGETS = {
    'sagittal': ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Contusion'],
    'coronal': ['MCL', 'Medial OA', 'Lateral OA'],
    'axial': ['PF OA', 'Effusion', 'Synovitis', "Baker's", 'Fracture']
}

PLANE_SLOTS = {
    'sagittal': [(0, 26, 8), (26, 48, 8)],     # 16 slices total
    'coronal': [(48, 66, 8), (66, 78, 8)],     # 16 slices total
    'axial': [(78, 96, 10), (0, 26, 6)]        # 16 slices total
}

def extract_plane_triples(vol_96, plane, res=224):
    """
    Extracts 2.5D slice triples (c-1, c, c+1) for the requested plane from a 96-slice Raptor volume (96, H, W).
    Matches train_engine5_specialists_kaggle.py exactly.
    Returns numpy array of shape (16, 3, res, res) in [0, 1].
    """
    import cv2
    slots = PLANE_SLOTS[plane]
    slabs = []
    
    for start, end, count in slots:
        max_c = end - 1
        steps = np.linspace(start, end - 1, count).round().astype(int)
        for c in steps:
            c = max(1, min(c, max_c - 1))
            tri = np.stack([vol_96[c - 1], vol_96[c], vol_96[c + 1]], axis=0).astype(np.float32) / 255.0 # (3, H, W)
            if tri.shape[1] != res or tri.shape[2] != res:
                # Resize each channel with INTER_BILINEAR
                chans = [cv2.resize(tri[ch], (res, res), interpolation=cv2.INTER_LINEAR) for ch in range(3)]
                tri = np.stack(chans, axis=0)
            slabs.append(tri)
            
    return np.stack(slabs[:16], axis=0)

def main():
    print("Testing Engine 5 inference logic...")
    dummy_vol = np.random.randint(0, 255, (96, 384, 384), dtype=np.uint8)
    
    for plane in ['sagittal', 'coronal', 'axial']:
        slabs = extract_plane_triples(dummy_vol, plane, res=224)
        targets = SPECIALIST_TARGETS[plane]
        print(f"Plane {plane:<10}: Slabs shape {slabs.shape} | Targets: {targets}")
        assert slabs.shape == (16, 3, 224, 224), f"Unexpected shape for {plane}: {slabs.shape}"
        assert len(targets) in [3, 4, 5], f"Unexpected targets count for {plane}: {len(targets)}"
    
    print("Engine 5 slice extraction logic validated successfully!")

if __name__ == '__main__':
    main()
