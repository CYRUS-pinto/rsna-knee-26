#!/usr/bin/env python3
"""
Overnight Continuous Loop Runner
Runs the optimization iteration every hour until stopped.
"""
import subprocess
import sys
import time
import signal
import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(r"C:\Users\Cyrus\Downloads\rsna knee fip").resolve()
EXP_DIR = REPO_ROOT / ".autoresearch" / "rsna-knee" / "overnight-optimization"
ITERATION_SCRIPT = EXP_DIR / "run_iteration.py"

# Global flag for graceful shutdown
shutdown = False

def signal_handler(sig, frame):
    global shutdown
    print(f"\n[{datetime.now().isoformat()}] Shutdown signal received, finishing current iteration...")
    shutdown = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_iteration():
    """Run one iteration of the optimization"""
    try:
        result = subprocess.run(
            [sys.executable, str(ITERATION_SCRIPT)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max per iteration
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Iteration timed out after 1 hour"
    except Exception as e:
        return False, "", str(e)

def main():
    print("=" * 60)
    print("OVERNIGHT CONTINUOUS LOOP STARTED")
    print("=" * 60)
    print(f"Repo: {REPO_ROOT}")
    print(f"Experiment: {EXP_DIR}")
    print(f"Interval: 1 hour")
    print(f"Started: {datetime.now().isoformat()}")
    print("Press Ctrl+C to stop gracefully")
    print("=" * 60)
    
    iteration = 0
    while not shutdown:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"LOOP ITERATION #{iteration} - {datetime.now().isoformat()}")
        print(f"{'='*60}")
        
        success, stdout, stderr = run_iteration()
        
        if success:
            print(f"Iteration {iteration} completed successfully")
            if "PASSED" in stdout:
                print("   Verification: PASSED")
            else:
                print("   Verification: Check output")
        else:
            print(f"Iteration {iteration} failed")
            if stderr:
                print(f"   Error: {stderr[:500]}")
            if stdout:
                print(f"   Output: {stdout[:500]}")
        
        if shutdown:
            print("\nShutdown requested, stopping...")
            break
        
        # Sleep for 1 hour (3600 seconds) but check shutdown every 60 seconds
        print(f"\n[{datetime.now().isoformat()}] Sleeping for 1 hour... (Ctrl+C to stop)")
        for _ in range(3600):
            if shutdown:
                break
            time.sleep(1)
    
    print("\n" + "=" * 60)
    print("OVERNIGHT LOOP STOPPED")
    print(f"Total iterations: {iteration}")
    print(f"Stopped: {datetime.now().isoformat()}")
    print("=" * 60)

if __name__ == "__main__":
    main()