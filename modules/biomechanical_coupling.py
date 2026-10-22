"""
Biomechanical Joint Kinematic Interaction & Coupling Module.
Applies orthopedic biomechanical laws in rank space:
1. Acute Trauma Coupling: Lateral femoral/tibial bone contusion + joint effusion -> ACL rupture elevation.
2. Degenerative Compartment Coupling:
   - Medial Osteoarthritis -> Medial Meniscus degradation.
   - Lateral Osteoarthritis -> Lateral Meniscus degradation.
Operates safely on percentile ranks [0, 1] without destroying monotonic calibration.
"""

import numpy as np
import pandas as pd

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def apply_biomechanical_coupling(df_ranks: pd.DataFrame, alpha_acl: float = 0.08, alpha_meniscus: float = 0.05) -> pd.DataFrame:
    """
    Modulates rank-standardized predictions using joint kinematic constraints.
    
    Args:
        df_ranks: DataFrame containing percentile ranks [0, 1] for all 12 pathologies.
        alpha_acl: Coupling strength for acute trauma -> ACL.
        alpha_meniscus: Coupling strength for OA -> Meniscal degeneration.
        
    Returns:
        coupled_df: Bounded DataFrame [0.0001, 0.9999] with coupled signals.
    """
    out_df = df_ranks.copy()
    
    # 1. Acute Trauma Kinematics (ACL)
    if 'ACL' in out_df.columns and 'Contusion' in out_df.columns and 'Effusion' in out_df.columns:
        p_acl = out_df['ACL'].values
        p_cont = out_df['Contusion'].values
        p_eff = out_df['Effusion'].values
        
        # When both contusion and effusion are high, secondary signs point to acute ligamentous injury
        trauma_signal = p_cont * p_eff
        acl_modulated = p_acl + alpha_acl * (trauma_signal * (1.0 - p_acl))
        out_df['ACL'] = np.clip(acl_modulated, 0.0001, 0.9999)
        
    # 2. Medial Compartment Degeneration
    if 'Medial Meniscus' in out_df.columns and 'Medial OA' in out_df.columns:
        p_mmen = out_df['Medial Meniscus'].values
        p_moa = out_df['Medial OA'].values
        mmen_modulated = p_mmen + alpha_meniscus * (p_moa * (1.0 - p_mmen))
        out_df['Medial Meniscus'] = np.clip(mmen_modulated, 0.0001, 0.9999)
        
    # 3. Lateral Compartment Degeneration
    if 'Lateral Meniscus' in out_df.columns and 'Lateral OA' in out_df.columns:
        p_lmen = out_df['Lateral Meniscus'].values
        p_loa = out_df['Lateral OA'].values
        lmen_modulated = p_lmen + alpha_meniscus * (p_loa * (1.0 - p_lmen))
        out_df['Lateral Meniscus'] = np.clip(lmen_modulated, 0.0001, 0.9999)
        
    return out_df

if __name__ == '__main__':
    data = {c: np.random.uniform(0.1, 0.9, 10) for c in TARGETS}
    df = pd.DataFrame(data)
    res = apply_biomechanical_coupling(df)
    assert res.shape == (10, 12), "Shape changed!"
    assert (res.values >= 0.0001).all() and (res.values <= 0.9999).all(), "Values out of bounds!"
    print("[+] Biomechanical Coupling unit test passed perfectly!")
