"""
Bayesian Pseudo-Label Target Harmonizer & Synovitis Prior Imputation.
Implements Pillar 1 of ADR-013:
- Overcomes narrative report noise where 44% of true synovitis cases are unmentioned in text.
- Keeps 58 radiologist-verified double-read gold consensus studies at 100% ground truth fidelity.
- Harmonizes 4,349 unannotated studies using teacher-student Bayesian visual predictions:
    Y_harmonized = alpha * Y_visual_oof + (1 - alpha) * Y_text
- Imputes conditioned clinical prior for unmentioned synovitis based on effusion and popliteal fluid.
"""

import numpy as np
import pandas as pd

ALL_LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def harmonize_training_targets(df_text, df_visual_oof, alpha=0.25):
    """
    Combines text extracted labels with out-of-fold visual predictions.
    
    Args:
        df_text: DataFrame containing 'StudyInstanceUID', 12 pathology columns, and optional 'is_gold'.
        df_visual_oof: DataFrame containing 'StudyInstanceUID' and 12 visual out-of-fold probability columns.
        alpha: Weight given to visual teacher predictions (default 0.25 for conservative regularization).
    Returns:
        df_harmonized: DataFrame with cleaned, continuous probability targets.
    """
    df_h = df_text.copy()
    oof_indexed = df_visual_oof.set_index('StudyInstanceUID')
    
    is_gold_mask = df_h.get('is_gold', pd.Series(0, index=df_h.index)).astype(bool)
    
    for col in ALL_LABELS:
        if col not in df_h.columns or col not in oof_indexed.columns:
            continue
            
        text_vals = df_h[col].values.astype(np.float32)
        uids = df_h['StudyInstanceUID'].values
        vis_vals = oof_indexed.loc[uids, col].values.astype(np.float32)
        
        # Harmonize targets for non-gold studies
        harmonized_col = np.where(
            is_gold_mask,
            text_vals, # Preserve exact gold truth for the 58 verified studies
            (1.0 - alpha) * text_vals + alpha * vis_vals
        )
        
        # Synovitis Prior Imputation:
        # If unmentioned in text (text == 0.0), impute conditioned prior when effusion and visual signal are positive
        if col == 'Synovitis' and 'Effusion' in df_h.columns:
            eff_vis = oof_indexed.loc[uids, 'Effusion'].values.astype(np.float32)
            syn_imputed = np.clip(0.35 * eff_vis + 0.65 * vis_vals, 0.0, 1.0)
            
            # Apply imputation only where text was negative and study is non-gold
            should_impute = (~is_gold_mask) & (text_vals < 0.1) & (vis_vals > 0.45)
            harmonized_col = np.where(should_impute, np.maximum(harmonized_col, syn_imputed), harmonized_col)
            
        df_h[col] = np.clip(harmonized_col, 0.0, 1.0)
        
    return df_h
