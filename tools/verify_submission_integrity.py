import sys
import pandas as pd
import numpy as np
from pathlib import Path

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]

def verify_submission(sub_path):
    p = Path(sub_path)
    if not p.exists():
        print(f"[FAIL] Submission file does not exist: {p}")
        return False
    
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[FAIL] Could not parse CSV: {e}")
        return False

    print(f"[PASS] Successfully loaded {p.name} with shape {df.shape}")

    # Check columns
    expected_cols = ["StudyInstanceUID"] + LABELS
    if list(df.columns) != expected_cols:
        print(f"[FAIL] Columns mismatch!\nExpected: {expected_cols}\nFound: {list(df.columns)}")
        return False
    print("[PASS] Columns match expected schema exactly.")

    # Check NaNs and Infs
    vals = df[LABELS].values
    if not np.isfinite(vals).all():
        nan_count = np.isnan(vals).sum()
        inf_count = np.isinf(vals).sum()
        print(f"[FAIL] Non-finite values detected! NaNs: {nan_count}, Infs: {inf_count}")
        return False
    print("[PASS] Zero NaNs or Infs detected.")

    # Check bounds [0.0, 1.0]
    if (vals < 0.0).any() or (vals > 1.0).any():
        print(f"[FAIL] Probabilities out of bounds [0.0, 1.0]! Min: {vals.min()}, Max: {vals.max()}")
        return False
    print(f"[PASS] All values strictly bounded within [0.0, 1.0] (Min: {vals.min():.4f}, Max: {vals.max():.4f})")

    # Check non-degeneracy (std > 1e-6 if rows > 1)
    if len(df) > 1:
        stds = df[LABELS].std()
        degenerate = stds[stds < 1e-6]
        if len(degenerate) > 0:
            print(f"[WARN] Degenerate/constant columns detected: {degenerate.to_dict()}")
        else:
            print("[PASS] All finding distributions are active and non-degenerate.")

    # Check UID uniqueness
    uids = df["StudyInstanceUID"].astype(str)
    if len(uids) != len(uids.unique()):
        print(f"[FAIL] Duplicate StudyInstanceUIDs found!")
        return False
    print(f"[PASS] All {len(uids)} StudyInstanceUIDs are unique.")

    print(f"\n[SUMMARY] Submission {p.name} PASSED all integrity checks!")
    return True

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "submissions/v17_multiscale_r6_0958/submission.csv"
    success = verify_submission(target)
    sys.exit(0 if success else 1)
