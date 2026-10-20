"""
Pre-Flight Health & Verification Audit for Final Selections
RSNA Knee Abnormality Detection Challenge 2026
"""
import pandas as pd
import numpy as np
from pathlib import Path

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def verify_submission(csv_path: Path, name: str):
    print(f"\nVerifying {name}: {csv_path}")
    assert csv_path.is_file(), f"File {csv_path} does not exist!"
    df = pd.read_csv(csv_path)
    
    assert 'StudyInstanceUID' in df.columns, "StudyInstanceUID column missing!"
    for target in TARGETS:
        assert target in df.columns, f"Target column {target} missing!"
        
    vals = df[TARGETS].values
    assert np.isfinite(vals).all(), "NaN or Inf detected in predictions!"
    assert (vals >= 0.0).all() and (vals <= 1.0).all(), "Values out of bounds [0, 1]!"
    
    print(f"  Shape: {df.shape}")
    print(f"  Unique studies: {df['StudyInstanceUID'].nunique()}")
    print(f"  Range: [{vals.min():.4f}, {vals.max():.4f}]")
    print(f"  Mean across all targets: {vals.mean():.4f}")
    print(f"  Std across all targets: {vals.std():.4f}")
    print(f"  STATUS: PASS")
    return df

if __name__ == '__main__':
    v17_path = Path("submissions/v17_multiscale_r6_0958/output/submission.csv")
    v6_path = Path("submissions/v06_pb_baseline/output/submission.csv")
    
    v17 = verify_submission(v17_path, "Slot 1 Anchor: Candidate V17 (0.958)")
    v6 = verify_submission(v6_path, "Slot 2 Hedge: Candidate V6 (0.956)")
    
    print("\n" + "="*50)
    print("ALL INTEGRITY AND SAFETY CHECKS PASSED FOR FINALS.")
    print("="*50)
