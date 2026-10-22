#!/usr/bin/env python3
"""
Overnight Optimization - Iteration Runner
Runs one iteration of the autoresearch experiment.
"""
import subprocess
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

EXP_DIR = Path(__file__).parent.resolve()
# Explicitly set repo root to handle spaces in path
REPO_ROOT = Path(r"C:\Users\Cyrus\Downloads\rsna knee fip").resolve()

def run_cmd(cmd, cwd=None, timeout=300):
    """Run command and return (success, stdout, stderr)"""
    working_dir = str((cwd or REPO_ROOT).resolve())
    print(f"  [DEBUG] Running: {cmd} in {working_dir}")
    try:
        # Use shell=True with explicit cwd for Windows
        result = subprocess.run(cmd, cwd=working_dir, capture_output=True, text=True, timeout=timeout, shell=True)
        print(f"  [DEBUG] Exit code: {result.returncode}")
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return False, "", str(e)

def log_step(action, description, result, status, notes=""):
    """Append to results.tsv"""
    timestamp = datetime.utcnow().isoformat() + "Z"
    with open(EXP_DIR / "results.tsv", "a") as f:
        f.write(f"{len(open(EXP_DIR / 'results.tsv').readlines())}\t{datetime.utcnow().isoformat()}Z\t{action}\t{description}\t\t{status}\t{notes}\n")

def main():
    print("=" * 60)
    print("OVERNIGHT OPTIMIZATION - ITERATION START")
    print("=" * 60)
    
    # Read current state
    with open(EXP_DIR / "config.cfg") as f:
        config = dict(line.strip().split(" = ") for line in f if "=" in line)
    
    target = config.get("target", "")
    evaluate_cmd = config.get("evaluate_cmd", "")
    
    print(f"Target: {target}")
    print(f"Evaluate: {evaluate_cmd}")
    print(f"Time: {datetime.utcnow().isoformat()}Z")
    
    # Determine next action based on run count
    results_file = EXP_DIR / "results.tsv"
    run_count = len(open(results_file).readlines()) - 1  # minus header
    
    print(f"Run #{run_count + 1}")
    
    if run_count == 0:
        # Run 1: OOF Extraction + Weight Optimization
        print("\n>>> ACTION: OOF Extraction + Nelder-Mead Optimization")
        print("This requires Colab GPU for Engine 12 (A100) and Engine 13 (T4)")
        print("Run these notebooks manually on Colab:")
        print("  TAB 1 (A100): colab_oof_extraction/engine12_13_fold0_oof_extraction.ipynb -> Cells 1-6")
        print("  TAB 2 (T4):   Same notebook -> Cells 1-5, 7")
        print("\nAfter completion, OOF CSVs will be pushed to Kaggle datasets.")
        print("Then run: python tools/run_oof_optimization.py")
        
        log_step("oof_extraction", "OOF extraction initiated on Colab", "", "INITIATED", "Waiting for Colab completion")
        
    elif run_count == 1:
        # Run 2: OOF Weight Optimization
        print("\n>>> ACTION: OOF Weight Optimization (Nelder-Mead)")
        success, out, err = run_cmd("python tools/run_oof_optimization.py")
        if success:
            log_step("oof_optimization", "Nelder-Mead per-pathology optimization", "weights injected", "SUCCESS", out[:200])
        else:
            log_step("oof_optimization", "Nelder-Mead optimization failed", "", "FAILED", err[:200])
            
    elif run_count == 2:
        # Run 3: Guardrail validation
        print("\n>>> ACTION: Ceiling Verification Suite")
        success, out, err = run_cmd("python tools/ceiling_verification_suite.py --quick")
        if success and "ALL CRITICAL TESTS PASSED" in out:
            log_step("verification", "Ceiling verification suite", "PASSED", "SUCCESS", "All 11 tests passed")
        else:
            log_step("verification", "Ceiling verification", "", "FAILED", err[:200] if err else out[:200])
            
    elif run_count == 3:
        # Run 4: Submit V49
        print("\n>>> ACTION: Submit V49 to Kaggle")
        target_path = "submissions/v49_oof_calibrated_quad_arm"
        success, out, err = run_cmd(f"kaggle kernels push -p {target_path}")
        if success:
            log_step("submit", "V49 submitted to Kaggle", "submitted", "SUCCESS", out[:200])
        else:
            log_step("submit", "V49 submission failed", "", "FAILED", err[:200])
            
    elif run_count >= 4 and run_count <= 8:
        # Runs 5-9: Multi-fold training
        print(f"\n>>> ACTION: Multi-fold Training (Run {run_count + 1})")
        print("Launch Colab tabs for parallel training:")
        print("  Engine 12: colab_folds/engine12_convnext_large/engine12_multi_fold_master.ipynb")
        print("  Engine 13: colab_folds/engine13_swin_base/engine13_multi_fold_master.ipynb")
        log_step("training", f"Multi-fold training initiated (run {run_count + 1})", "", "INITIATED", "Colab parallel training")
        
    elif run_count >= 9:
        # Runs 10+: Advanced architecture
        print(f"\n>>> ACTION: Advanced Architecture (Run {run_count + 1})")
        log_step("architecture", f"Advanced architecture work (run {run_count + 1})", "", "PLANNED", "DINOv2, cross-planar, TTA, etc.")
        
    else:
        print("Unknown run state")
        return 1
    
    # Run verification
    print("\n>>> Running ceiling verification...")
    success, out, err = run_cmd("python tools/ceiling_verification_suite.py --quick")
    if "ALL CRITICAL TESTS PASSED" in out:
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        print(err[:500] if err else out[:500])
    
    # Update state snapshot
    run_cmd("python tools/state_snapshot.py")
    
    print("\n" + "=" * 60)
    print("ITERATION COMPLETE")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())