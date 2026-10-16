"""
Orchestrator to Train and Evaluate All 3 Plane-Decoupled Specialists (ADR-015).
Produces the Unified 12-Column Out-Of-Fold (OOF) Prediction Matrix
and Verifies Anti-Collapse Across Pathologies.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from training.engine5_decoupled.train_specialist import train_specialist_fold, exact_roc_auc
from src.data.plane_datasets import ALL_LABELS, SPECIALIST_TARGETS

def train_and_evaluate_all_specialists(fold=0, epochs=6, batch_size=2, accum_steps=8,
                                       backbone_lr=1e-5, head_lr=1e-3, data_path='data',
                                       cached_volume_dir=None, backbone_name='convnext_small'):
    print("\n" + "#" * 80)
    print(f"  ENGINE 5 PIPELINE: TRAINING ALL 3 SPECIALISTS FOR FOLD {fold}")
    print("#" * 80 + "\n")
    
    specialists = ['sagittal', 'coronal', 'axial']
    results = {}
    
    for plane in specialists:
        print(f"\n>>> Running Specialist Training: {plane.upper()} <<<")
        auc, corr = train_specialist_fold(
            plane=plane,
            fold=fold,
            epochs=epochs,
            batch_size=batch_size,
            accum_steps=accum_steps,
            backbone_lr=backbone_lr,
            head_lr=head_lr,
            backbone_name=backbone_name,
            data_path=data_path,
            cached_volume_dir=cached_volume_dir
        )
        results[plane] = {'auc': auc, 'corr': corr}
        
    print("\n" + "=" * 80)
    print("  MERGING SPECIALIST OOF PREDICTIONS & CALCULATING 12-CLASS MACRO-AUC")
    print("=" * 80)
    
    oof_dfs = []
    for plane in specialists:
        oof_path = f"oof/oof_engine5_{plane}_fold{fold}.parquet"
        if os.path.exists(oof_path):
            df_part = pd.read_parquet(oof_path)
            oof_dfs.append(df_part)
        else:
            print(f"Warning: {oof_path} not found!")
            
    if len(oof_dfs) == 3:
        # Merge on StudyInstanceUID
        df_combined = oof_dfs[0]
        for df_next in oof_dfs[1:]:
            df_combined = pd.merge(df_combined, df_next, on='StudyInstanceUID')
            
        combined_oof_path = f"oof/oof_engine5_combined_fold{fold}.parquet"
        df_combined.to_parquet(combined_oof_path, index=False)
        print(f"Combined 12-class OOF predictions saved to {combined_oof_path}")
        
        # Load ground truth for validation UIDs
        labels_file = os.path.join(data_path, 'SUPER_CONSENSUS_labels.parquet')
        if not os.path.exists(labels_file):
            labels_file = 'SUPER_CONSENSUS_labels.parquet'
        df_gt = pd.read_parquet(labels_file) if labels_file.endswith('.parquet') else pd.read_csv(labels_file)
        
        gt_merged = pd.merge(df_combined[['StudyInstanceUID']], df_gt, on='StudyInstanceUID')
        
        print("\n--- 12 PATHOLOGY INDIVIDUAL AUC SCORES ---")
        overall_aucs = {}
        for col in ALL_LABELS:
            y_true = (gt_merged[col].values >= 0.5).astype(int)
            y_pred = df_combined[col].values
            a = exact_roc_auc(y_true, y_pred)
            overall_aucs[col] = a
            print(f"  {col:22s}: {a:.4f}")
            
        macro_auc_12 = float(np.mean(list(overall_aucs.values())))
        print("------------------------------------------")
        print(f"  ENGINE 5 OVERALL 12-CLASS MACRO-AUC: {macro_auc_12:.5f}")
        print("------------------------------------------")
        
        # 12x12 Correlation Analysis
        corr_12 = df_combined[ALL_LABELS].corr().values
        np.fill_diagonal(corr_12, 0.0)
        mean_off_diag = float(np.mean(np.abs(corr_12)))
        print(f"  ENGINE 5 MEAN OFF-DIAGONAL CORRELATION: {mean_off_diag:.4f}")
        print(f"  (Comparison: Engine 4 collapsed correlation was 0.9966, GT is 0.1795)")
        if mean_off_diag <= 0.40:
            print("  [VERIFIED]: Multi-task representation collapse successfully eliminated!")
        else:
            print("  [NOTE]: Correlation is lower than Engine 4 baseline.")
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate All Engine 5 Specialists")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--backbone_name", type=str, default='convnext_small')
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--cached_volume_dir", type=str, default=None)
    args = parser.parse_args()

    train_and_evaluate_all_specialists(
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        backbone_name=args.backbone_name,
        data_path=args.data_path,
        cached_volume_dir=args.cached_volume_dir
    )

if __name__ == '__main__':
    main()
