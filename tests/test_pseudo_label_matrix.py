import numpy as np
import pandas as pd
from pathlib import Path

def test_pseudo_label_matrix_properties():
    target_path = Path("data/train_folds.parquet")
    assert target_path.is_file(), f"Target file {target_path} missing!"
    
    df = pd.read_parquet(target_path)
    assert df.shape == (4407, 15), f"Unexpected shape: {df.shape}"
    
    # 1. No NaNs
    assert not df.isna().any().any(), "Found NaNs in target matrix!"
    
    # 2. Key columns
    targets = [
        'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
        'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
        'Synovitis', "Baker's", 'Contusion', 'Fracture'
    ]
    for c in targets:
        assert c in df.columns, f"Target {c} missing!"
        vals = df[c].values
        # Probabilities bounded in [0, 1]
        assert vals.min() >= 0.0 and vals.max() <= 1.0, f"{c} out of [0, 1] bounds"
        # Non-trivial variance
        assert vals.std() > 0.05, f"{c} collapsed! std={vals.std()}"
        
    # 3. Gold subset
    assert "is_gold" in df.columns
    assert df["is_gold"].sum() == 58, f"Expected 58 gold studies, got {df['is_gold'].sum()}"
    
    # 4. Folds
    assert "fold" in df.columns
    assert set(df["fold"].unique()) == {0, 1, 2, 3, 4}
    for f in range(5):
        n_fold = (df["fold"] == f).sum()
        assert 870 <= n_fold <= 895, f"Fold {f} has unbalanced size: {n_fold}"
        
    print("PASS: Target matrix passed all integrity, fold, and distribution checks.")

if __name__ == "__main__":
    test_pseudo_label_matrix_properties()
