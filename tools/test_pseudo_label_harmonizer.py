import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
from src.data.pseudo_label_harmonizer import harmonize_training_targets, ALL_LABELS

def test_harmonization():
    print("Testing harmonize_training_targets...")
    
    # Study 0: Gold study (must remain 100% untouched)
    # Study 1: Regular non-gold study (smooth blend)
    # Study 2: Unmentioned synovitis with strong effusion/visual signal (imputed prior)
    df_text = pd.DataFrame({
        'StudyInstanceUID': ['uid_gold', 'uid_normal', 'uid_syn_omitted'],
        'is_gold': [1, 0, 0],
        'ACL': [1.0, 0.0, 0.5],
        'Effusion': [0.5, 0.2, 0.9],
        'Synovitis': [0.0, 0.0, 0.0], # Synovitis omitted in text for study 2
        'Medial Meniscus': [0.5, 0.5, 0.5]
    })
    
    df_vis = pd.DataFrame({
        'StudyInstanceUID': ['uid_gold', 'uid_normal', 'uid_syn_omitted'],
        'ACL': [0.2, 0.8, 0.5],
        'Effusion': [0.8, 0.1, 0.95],
        'Synovitis': [0.9, 0.05, 0.85], # Visual model detects 0.85 synovitis in study 2
        'Medial Meniscus': [0.1, 0.9, 0.5]
    })
    
    df_h = harmonize_training_targets(df_text, df_vis, alpha=0.25)
    
    # 1. Gold study check: ACL should remain exactly 1.0 (not blended with 0.2)
    assert df_h.loc[0, 'ACL'] == 1.0, f"Gold study was modified! {df_h.loc[0, 'ACL']}"
    assert df_h.loc[0, 'Synovitis'] == 0.0, "Gold study synovitis was modified!"
    print("Gold invariance passed!")
    
    # 2. Normal study check: ACL was 0.0, visual was 0.8, blend alpha=0.25 -> 0.2
    expected_acl = 0.75 * 0.0 + 0.25 * 0.8 # 0.2
    assert abs(df_h.loc[1, 'ACL'] - expected_acl) < 1e-4, f"Normal blend failed: {df_h.loc[1, 'ACL']}"
    print("Teacher-student blending passed!")
    
    # 3. Synovitis imputation check: Text was 0.0, visual was 0.85, effusion was 0.95 -> should be imputed > 0.5
    assert df_h.loc[2, 'Synovitis'] > 0.5, f"Synovitis was not imputed! {df_h.loc[2, 'Synovitis']}"
    print(f"Imputed Synovitis probability: {df_h.loc[2, 'Synovitis']:.4f}")
    print("All harmonization tests passed successfully!")

if __name__ == '__main__':
    test_harmonization()
