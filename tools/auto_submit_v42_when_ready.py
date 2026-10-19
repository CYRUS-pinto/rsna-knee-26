import os
import sys
import time
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def auto_submit_when_ready(kernel_slug='cyruspinto/rsna-knee-v42-sovereign-live-master', poll_interval=20, timeout_minutes=45):
    kaggle.api.authenticate()
    print(f"[*] Starting Auto-Submitter for {kernel_slug}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout_minutes * 60:
        try:
            status_res = kaggle.api.kernels_status(kernel_slug)
            curr_status = getattr(status_res, 'status', None)
            elapsed = int(time.time() - start_time)
            print(f"[{elapsed}s] Kernel Status: {curr_status}", flush=True)
            
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
            print("Poll error:", type(e), e, flush=True)
            
        time.sleep(poll_interval)
        
    print("[!] Timeout waiting for kernel commit.", flush=True)
    return False

if __name__ == '__main__':
    auto_submit_when_ready()
