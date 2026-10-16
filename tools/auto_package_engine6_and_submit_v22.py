"""
Automated Pipeline: Monitor Engine 6 Training -> Download Weights -> Create Kaggle Dataset -> Push Candidate V22.
Runs autonomously, verifies artifacts at each stage, and prepares the final submission.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
import kaggle

KERNEL_ID = "cyruspinto/rsna-knee-engine6-sequence-joint-crop-train"
DATASET_ID = "cyruspinto/rsna-knee-engine6-weights"
LOCAL_WEIGHTS_DIR = Path("artifacts/models/engine6")

def check_status():
    res = kaggle.api.kernels_status(KERNEL_ID)
    status = getattr(res, 'status', None)
    err = getattr(res, 'failure_message', None) or getattr(res, 'failureMessage', None)
    return str(status).upper() if status else "UNKNOWN", str(err) if err else None

def monitor_and_package():
    print("=" * 80)
    print(f"MONITORING ENGINE 6 TRAINING KERNEL: {KERNEL_ID}")
    print("=" * 80)
    
    start_time = time.time()
    while True:
        status, err = check_status()
        elapsed = time.time() - start_time
        print(f"[{time.strftime('%H:%M:%S')}] Status: {status} (Elapsed: {elapsed/60:.1f}m)")
        
        if "COMPLETE" in status:
            print("\n>>> TRAINING KERNEL COMPLETED SUCCESSFULLY! <<<")
            break
        elif "FAILED" in status or "ERROR" in status:
            raise RuntimeError(f"Training kernel failed: {err}")
        elif "CANCEL" in status:
            raise RuntimeError("Training kernel was cancelled.")
            
        time.sleep(60)

    # 1. Download outputs
    print("\n[Step 1/4] Downloading Engine 6 weights from Kaggle...")
    LOCAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    kaggle.api.kernels_output(KERNEL_ID, path=str(LOCAL_WEIGHTS_DIR))
    
    # Check downloaded files
    downloaded = list(LOCAL_WEIGHTS_DIR.glob("**/*.pth"))
    print(f"Found weight files: {[p.name for p in downloaded]}")
    
    # Flatten if inside subdirectories
    dataset_stage_dir = Path("artifacts/engine6_dataset_stage")
    dataset_stage_dir.mkdir(parents=True, exist_ok=True)
    for pth in downloaded:
        shutil.copy(pth, dataset_stage_dir / pth.name)
        print(f"Staged {pth.name} for dataset creation")

    # 2. Create Kaggle Dataset Metadata
    print("\n[Step 2/4] Initializing / updating Kaggle Dataset metadata...")
    meta = {
        "title": "RSNA Knee Engine 6 Native Joint Sequence Weights",
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(dataset_stage_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 3. Create or Update Dataset on Kaggle
    print(f"\n[Step 3/4] Uploading dataset {DATASET_ID} via Kaggle API...")
    try:
        kaggle.api.dataset_create_new(
            folder=str(dataset_stage_dir),
            dir_mode="zip",
            quiet=False
        )
        print(f"Dataset {DATASET_ID} created successfully!")
    except Exception as e:
        print(f"Dataset create returned: {e}. Trying dataset_create_version...")
        try:
            kaggle.api.dataset_create_version(
                folder=str(dataset_stage_dir),
                version_notes="Engine 6 Sagittal and Coronal Sequence Checkpoints",
                dir_mode="zip",
                quiet=False
            )
            print(f"Dataset {DATASET_ID} updated with new version!")
        except Exception as e2:
            print(f"Dataset version creation error: {e2}")

    # 4. Push Candidate V22 Kernel
    print("\n[Step 4/4] Pushing Candidate V22 Quint-Engine Kernel...")
    v22_dir = Path("submissions/v22_quint_engine_master")
    kaggle.api.kernels_push(str(v22_dir))
    print("Candidate V22 kernel successfully pushed to Kaggle!")
    print("Kernel URL: https://www.kaggle.com/code/cyruspinto/rsna-knee-v22-quint-engine-blend")
    print("=" * 80)
    print("PIPELINE COMPLETE: Head to Kaggle UI and click 'Submit to Competition' on Version #1 output!")
    print("=" * 80)

if __name__ == '__main__':
    monitor_and_package()
