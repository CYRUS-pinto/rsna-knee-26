"""
Sagittal Acute Tear Specialist Training Entry Point (ADR-015).
Targets: ACL, Medial Meniscus, Lateral Meniscus, Bone Contusion.
Loss: Asymmetric Focal BCE (gamma=2, alpha=0.75)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from training.engine5_decoupled.train_specialist import train_specialist_fold

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train Sagittal Specialist")
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

    train_specialist_fold(
        plane='sagittal',
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
