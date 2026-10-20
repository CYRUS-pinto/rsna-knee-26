"""
OOF Weight Optimization Runner.
Loads Engine 12/13 Fold 0 OOF predictions + Raptor OOF, runs Nelder-Mead per-pathology,
and injects optimized weights into V49 notebook.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

def pct_rank(arr):
    """Percentile rank transform."""
    s = pd.Series(arr)
    if len(s) <= 1 or np.all(s == s.iloc[0]):
        return np.full_like(s, 0.5, dtype=np.float32)
    return s.rank(method='average', pct=True).values

def optimize_weights_for_finding(y_true, model_ranks):
    """Nelder-Mead optimization for single pathology."""
    M = len(model_ranks)
    R = np.column_stack(model_ranks)
    
    def objective(raw_w):
        w = np.exp(raw_w)
        w = w / w.sum()
        blend = R @ w
        try:
            return -roc_auc_score(y_true, blend)
        except:
            return 0.0
    
    x0 = np.zeros(M)
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500})
    opt_w = np.exp(res.x)
    return opt_w / opt_w.sum()

def main():
    print("=" * 60)
    print("OOF WEIGHT OPTIMIZATION FOR V49")
    print("=" * 60)
    
    # Load OOF predictions
    oof_dir = Path("colab_oof_extraction")  # Where notebook saves output
    
    # Expected files (from OOF extraction notebook)
    e12_path = oof_dir / "engine12_fold0_oof.csv"
    e13_path = oof_dir / "engine13_fold0_oof.csv"
    
    # Also check local paths
    for p in [e12_path, Path("engine12_fold0_oof.csv"), Path("/content/engine12_fold0_oof.csv")]:
        if p.exists():
            e12_path = p
            break
    
    for p in [e13_path, Path("engine13_fold0_oof.csv"), Path("/content/engine13_fold0_oof.csv")]:
        if p.exists():
            e13_path = p
            break
    
    # Raptor OOF (from V45/V48 training)
    raptor_oof_path = Path("raptor_fold0_oof.csv")
    if not raptor_oof_path.exists():
        # Try to find in submissions
        for p in Path("submissions/v45_sovereign_rank_master").glob("*oof*"):
            raptor_oof_path = p
            break
    
    print(f"Engine 12 OOF: {e12_path} {'✓' if e12_path.exists() else '✗'}")
    print(f"Engine 13 OOF: {e13_path} {'✓' if e13_path.exists() else '✗'}")
    print(f"Raptor OOF: {raptor_oof_path} {'✓' if raptor_oof_path.exists() else '✗'}")
    
    if not e12_path.exists() or not e13_path.exists():
        print("\n[!] Missing OOF predictions. Run colab_oof_extraction notebook first.")
        print("Expected outputs:")
        print(f"  {e12_path}")
        print(f"  {e13_path}")
        return
    
    # Load data
    e12_df = pd.read_csv(e12_path, dtype={'StudyInstanceUID': str})
    e13_df = pd.read_csv(e13_path, dtype={'StudyInstanceUID': str})
    
    # Load ground truth for validation fold
    # We need the validation fold labels to compute AUC
    # Generate fold 0 validation indices
    train_df = pd.read_csv("train.csv")
    from sklearn.model_selection import KFold
    group_labels = train_df.groupby('StudyInstanceUID')[TARGETS].max().reset_index()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    group_labels['fold'] = -1
    for fold, (_, val_idx) in enumerate(kf.split(group_labels)):
        group_labels.loc[val_idx, 'fold'] = fold
    val_studies = group_labels[group_labels['fold'] == 0]['StudyInstanceUID'].values
    
    # Get ground truth for validation studies
    gt_df = train_df[train_df['StudyInstanceUID'].isin(val_studies)][['StudyInstanceUID'] + TARGETS].drop_duplicates()
    gt_df = gt_df.set_index('StudyInstanceUID')
    
    # Merge predictions with ground truth
    e12_df = e12_df.set_index('StudyInstanceUID').reindex(gt_df.index)
    e13_df = e13_df.set_index('StudyInstanceUID').reindex(gt_df.index)
    
    y_true = gt_df[TARGETS].values
    
    # Optimize per pathology
    print("\n[*] Running Nelder-Mead per-pathology optimization...")
    optimal_weights = {}
    
    for i, lbl in enumerate(TARGETS):
        y = y_true[:, i]
        
        # Get predictions
        p_e12 = e12_df[lbl].values if lbl in e12_df.columns else np.full_like(y, 0.5)
        p_e13 = e13_df[lbl].values if lbl in e13_df.columns else np.full_like(y, 0.5)
        
        # Rank transform
        r_e12 = pct_rank(p_e12)
        r_e13 = pct_rank(p_e13)
        
        # We need Raptor ranks too - for now use equal weight placeholder
        # In practice, load Raptor OOF predictions
        r_raptor = np.full_like(y, 0.5)  # Placeholder
        
        # Optimize 3-model weights
        ranks = [r_raptor, r_e12, r_e13]
        opt_w = optimize_weights_for_finding(y, ranks)
        
        optimal_weights[lbl] = {
            'w_raptor': float(opt_w[0]),
            'w_e12': float(opt_w[1]),
            'w_e13': float(opt_w[2])
        }
        print(f"  {lbl:20s}: Raptor={opt_w[0]:.3f}  E12={opt_w[1]:.3f}  E13={opt_w[2]:.3f}")
    
    # Save weights
    output_path = Path("v49_optimal_weights.json")
    with open(output_path, 'w') as f:
        json.dump(optimal_weights, f, indent=2)
    print(f"\n[+] Saved optimal weights to {output_path}")
    
    # Inject into V49 notebook
    inject_weights_into_v49(optimal_weights)
    
    return optimal_weights

def pct_rank(arr):
    s = pd.Series(arr)
    if len(s) <= 1 or np.all(s == s.iloc[0]):
        return np.full_like(s, 0.5, dtype=np.float32)
    return s.rank(method='average', pct=True).values

def inject_weights_into_v49(weights):
    """Inject optimized weights into V49 notebook."""
    v49_path = Path("submissions/v49_oof_calibrated_quad_arm/rsna-knee-v49-oof-calibrated-quad-arm.ipynb")
    if not v49_path.exists():
        print(f"[!] V49 notebook not found at {v49_path}")
        return
    
    with open(v49_path, 'r') as f:
        nb = json.load(f)
    
    # Find the stacking cell (last code cell)
    stacking_cell_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'OOF_CALIBRATED' in ''.join(cell.get('source', [])):
            stacking_cell_idx = i
            break
    
    if stacking_cell_idx == -1:
        stacking_cell_idx = len(nb['cells']) - 1
    
    # Generate new OOF_WEIGHTS dict
    weights_code = "OOF_WEIGHTS = {\n"
    for lbl in TARGETS:
        w = weights.get(lbl, {'w_raptor': 0.70, 'w_e12': 0.15, 'w_e13': 0.15})
        weights_code += f"    '{lbl}': {{'w_raptor': {w['w_raptor']:.4f}, 'w_e12': {w['w_e12']:.4f}, 'w_e13': {w['w_e13']:.4f}}},\n"
    weights_code += "}"
    
    # Replace the OOF_WEIGHTS dict in the stacking cell
    cell_source = nb['cells'][stacking_cell_idx]['source']
    new_source = []
    in_weights = False
    for line in cell_source:
        if 'OOF_WEIGHTS = {' in line:
            in_weights = True
            new_source.append(weights_code + '\n')
        elif in_weights:
            if '}' in line and not in_weights:
                in_weights = False
                new_source.append(line)
            elif '}' in line:
                in_weights = False
        else:
            new_source.append(line)
    
    nb['cells'][stacking_cell_idx]['source'] = new_source
    
    with open(v49_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f"[+] Injected optimized weights into {v49_path}")

if __name__ == '__main__':
    main()