import os
import sys
import time
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def auto_submit_when_ready(kernel_slug='cyruspinto/fork-of-rsna-knee-v42-sovereign-live-master', poll_interval=30, timeout_minutes=180):
    kaggle.api.authenticate()
    print(f"[*] Starting Auto-Submitter for {kernel_slug} (Timeout: {timeout_minutes}m, Interval: {poll_interval}s)...")
    start_time = time.time()
    
    consecutive_errors = 0
    while time.time() - start_time < timeout_minutes * 60:
        try:
            status_res = kaggle.api.kernels_status(kernel_slug)
            curr_status = getattr(status_res, 'status', None)
            elapsed = int(time.time() - start_time)
            print(f"[{elapsed}s] Kernel Status: {curr_status}", flush=True)
            consecutive_errors = 0
            
            if 'COMPLETE' in str(curr_status):
                print(f"[+] Kernel {kernel_slug} is COMPLETE! Submitting to Kaggle competition...", flush=True)
                res = kaggle.api.competition_submit_code(
                    file_name='submission.csv',
                    message='Candidate V42 Sovereign Live Master (Engine 11 4-Fold 0.8712 + Engine 9 + Engine 10 + Super-CoAtNet + R7 + Raptor)',
                    competition='rsna-knee-abnormality-detection',
                    kernel=kernel_slug,
                    kernel_version=1
                )
                print("[+] SUBMISSION SUCCESSFUL! Kaggle Response:", res, flush=True)
                return True
            elif 'ERROR' in str(curr_status):
                failure_msg = getattr(status_res, 'failure_message', 'No details')
                print(f"[!] Kernel {kernel_slug} failed with ERROR: {failure_msg}", flush=True)
                return False
        except Exception as e:
            consecutive_errors += 1
            print(f"Poll error ({consecutive_errors}):", type(e).__name__, str(e)[:120], flush=True)
            if consecutive_errors > 15:
                print("[!] Too many consecutive network errors. Exiting watcher.", flush=True)
                return False
            
        time.sleep(poll_interval)
        
    print("[!] Timeout waiting for kernel commit.", flush=True)
    return False

if __name__ == '__main__':
    auto_submit_when_ready()
