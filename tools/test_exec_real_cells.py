"""
Comprehensive dry run: execute the REAL notebook cell source code from both
rsna-knee-v32-tri-engine-medical-master.ipynb and rsna-knee-v32-ablation-stride2-norad.ipynb
using synthetic data, directly testing the actual cells shipped in the notebooks.
"""

import json
import time
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

def test_notebook_cells(nb_path, is_ablation=False):
    print(f"\n=======================================================")
    print(f"Testing real cells from: {nb_path.name}")
    print(f"=======================================================")
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    temp_dir = Path(tempfile.mkdtemp())
    try:
        data_dir = temp_dir / "input"
        work_dir = temp_dir / "working"
        data_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Create synthetic test set (10 studies)
        test_ids = [f"test_{i:04d}" for i in range(10)]
        test_df = pd.DataFrame({"StudyInstanceUID": test_ids})
        test_df.to_csv(data_dir / "test.csv", index=False)
        
        labels = [
            "ACL", "Baker's", "Contusion", "Effusion", "Fracture",
            "Lateral Meniscus", "Lateral OA", "MCL",
            "Medial Meniscus", "Medial OA", "PF OA", "Synovitis"
        ]
        
        ss_df = pd.DataFrame(0.5, index=range(10), columns=["StudyInstanceUID"] + labels)
        ss_df["StudyInstanceUID"] = test_ids
        ss_df.to_csv(data_dir / "sample_submission.csv", index=False)
        
        # 2. Create synthetic Raptor and ConvNeXt model outputs in WORK
        np.random.seed(42)
        raw_vals = np.random.uniform(0.1, 0.9, size=(10, len(labels)))
        raw_df = pd.DataFrame(raw_vals, columns=labels)
        raw_df.insert(0, "StudyInstanceUID", test_ids)
        raw_df.to_csv(work_dir / "raptor_raw.csv", index=False)
        raw_df.to_csv(work_dir / "raptor_sub.csv", index=False)
        
        raw224_vals = np.random.uniform(0.1, 0.9, size=(10, len(labels)))
        raw224_df = pd.DataFrame(raw224_vals, columns=labels)
        raw224_df.insert(0, "StudyInstanceUID", test_ids)
        raw224_df.to_csv(work_dir / "raptor224_raw.csv", index=False)
        raw224_df.to_csv(work_dir / "raptor224_sub.csv", index=False)
        
        ours_vals = np.random.uniform(0.1, 0.9, size=(10, len(labels)))
        ours_df = pd.DataFrame(ours_vals, columns=labels)
        ours_df.insert(0, "StudyInstanceUID", test_ids)
        ours_df.to_csv(work_dir / "ours_sub.csv", index=False)
        
        # Create mock worker prediction files for ConvNeXt
        pred_dir = work_dir / "prediction"
        worker_dir = pred_dir / "worker0"
        worker_dir.mkdir(parents=True, exist_ok=True)
        ours_df.to_csv(worker_dir / "predictions.csv", index=False)
        
        # Create rad_raw.csv with non-trivial std
        rad_vals = np.random.uniform(0.2, 0.8, size=(10, len(labels)))
        rad_df = pd.DataFrame(rad_vals, columns=labels)
        rad_df.insert(0, "StudyInstanceUID", test_ids)
        rad_df.to_csv(work_dir / "rad_raw.csv", index=False)
        
        # Dynamically locate the Stage 3 ensembling cell (the final cell producing submission.csv)
        fusion_cell_idx = None
        for idx in reversed(range(len(nb["cells"]))):
            src = "".join(nb["cells"][idx].get("source", []))
            if "submission.csv" in src and "to_csv" in src and not src.strip().startswith("%%"):
                fusion_cell_idx = idx
                break
        assert fusion_cell_idx is not None, "Could not find submission generation cell in notebook!"
        
        cell_src = "".join(nb["cells"][fusion_cell_idx]["source"])
        
        # Setup mock global namespace matching Kaggle notebook environment at ensembling cell
        mock_globals = {
            "__name__": "__main__",
            "DATA": data_dir,
            "WORK": work_dir,
            "pred": pred_dir,
            "NB_T0": time.time() - 300.0,
            "RAPTOR_SECONDS": 45.0,
            "RAPTOR224_SECONDS": 25.0,
            "pd": pd,
            "np": np,
            "time": time,
            "json": json,
            "display": lambda x: print("display() called successfully on shape:", x.shape),
        }
        
        print(f"Executing Cell {fusion_cell_idx} source directly via exec()...")
        exec(cell_src, mock_globals)
        
        # Verify output submission file
        sub_path = work_dir / "submission.csv"
        assert sub_path.is_file(), "submission.csv was not produced!"
        out_sub = pd.read_csv(sub_path)
        print(f"Output submission.csv verified: shape {out_sub.shape}")
        assert list(out_sub.columns) == ["StudyInstanceUID"] + labels
        assert out_sub["StudyInstanceUID"].tolist() == test_ids
        assert not out_sub.isna().any().any(), "Found NaNs in submission.csv!"
        assert (out_sub[labels].values >= 0.0).all() and (out_sub[labels].values <= 1.0).all(), "Values out of bounds!"
        
        # Check receipt
        receipt_files = list(work_dir.glob("receipt_*.json"))
        if receipt_files:
            receipt = json.loads(receipt_files[0].read_text())
            print(f"Receipt written: {receipt.get('mode', 'N/A')}, elapsed: {receipt.get('total_seconds', 'N/A')}s")
        print(f"Verification PASSED for {nb_path.name} (Cell {fusion_cell_idx})!")
        
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
        assert target_path.is_file(), f"Target notebook {target_path} does not exist!"
        test_notebook_cells(target_path)
    else:
        v31_path = Path("submissions/v31_stride2_capacity_master/rsna-knee-v31-stride2-capacity-master.ipynb")
        if v31_path.is_file():
            test_notebook_cells(v31_path)
        
        v32_path = Path("submissions/v32_tri_engine_medical_master/rsna-knee-v32-tri-engine-medical-master.ipynb")
        if v32_path.is_file():
            test_notebook_cells(v32_path)
        
        ablation_path = Path("submissions/v32_ablation_stride2_norad/rsna-knee-v32-ablation-stride2-norad.ipynb")
        if ablation_path.is_file():
            test_notebook_cells(ablation_path, is_ablation=True)
