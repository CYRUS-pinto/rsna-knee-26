"""
Auto-submitter and Monitor for Candidate V45 Sovereign Rank Master.
Monitors cyruspinto/rsna-knee-v45-sovereign-rank-master until COMPLETE,
then submits to the RSNA Knee competition using today's fresh quota (Oct 20 UTC).
After submitting, monitors the competition submission status and logs the Public LB score.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv('.env')
import kaggle

def auto_submit_and_monitor(kernel_slug='cyruspinto/rsna-knee-v45-sovereign-rank-master', poll_interval=30, timeout_minutes=240):
    kaggle.api.authenticate()
    print(f"[*] Starting Auto-Submitter for {kernel_slug} (Timeout: {timeout_minutes}m, Interval: {poll_interval}s)...", flush=True)
    start_time = time.time()
    
    consecutive_errors = 0
    submitted = False
    sub_ref = None
    
    while time.time() - start_time < timeout_minutes * 60:
        try:
            if not submitted:
                status_res = kaggle.api.kernels_status(kernel_slug)
                curr_status = getattr(status_res, 'status', None)
                elapsed = int(time.time() - start_time)
                print(f"[{elapsed}s] Kernel Status: {curr_status}", flush=True)
                consecutive_errors = 0
                
                if 'COMPLETE' in str(curr_status):
                    print(f"[+] Kernel {kernel_slug} is COMPLETE! Submitting to Kaggle competition...", flush=True)
                    res = kaggle.api.competition_submit_code(
                        file_name='submission.csv',
                        message='Candidate V45 Sovereign Rank Master (Percentile Rank Ensembling & Biomechanical Coupling)',
                        competition='rsna-knee-abnormality-detection',
                        kernel=kernel_slug,
                        kernel_version=1
                    )
                    print("[+] SUBMISSION SUCCESSFUL! Kaggle Response:", res, flush=True)
                    submitted = True
                    # Retrieve the submission reference
                    time.sleep(10)
                    subs = kaggle.api.competition_submissions('rsna-knee-abnormality-detection')
                    if subs:
                        sub_ref = str(subs[0].ref)
                        print(f"[*] Tracked new competition submission Ref: {sub_ref}", flush=True)
                elif 'ERROR' in str(curr_status):
                    failure_msg = getattr(status_res, 'failure_message', 'No details')
                    print(f"[!] Kernel {kernel_slug} failed with ERROR: {failure_msg}", flush=True)
                    return False
            else:
                # Monitor the competition scoring pipeline
                elapsed = int(time.time() - start_time)
                subs = kaggle.api.competition_submissions('rsna-knee-abnormality-detection')
                target_sub = next((s for s in subs if str(s.ref) == sub_ref), None) if sub_ref else subs[0]
                
                if target_sub:
                    sub_status = getattr(target_sub, 'status', 'UNKNOWN')
                    score = getattr(target_sub, 'public_score', None)
                    print(f"[{elapsed}s] Submission {target_sub.ref} Status: {sub_status} | Score: {score}", flush=True)
                    
                    if 'COMPLETE' in str(sub_status):
                        print("==================================================", flush=True)
                        print(f"[+] CANDIDATE V45 SCORING COMPLETE! Public Score: {score}", flush=True)
                        print("==================================================", flush=True)
                        return True
                    elif 'ERROR' in str(sub_status):
                        err_desc = getattr(target_sub, 'error_description', 'No details')
                        print(f"[!] Submission failed with error: {err_desc}", flush=True)
                        return False
                        
        except Exception as e:
            consecutive_errors += 1
            print(f"Poll error ({consecutive_errors}):", type(e).__name__, str(e)[:120], flush=True)
            if consecutive_errors > 20:
                print("Too many consecutive errors, aborting.", flush=True)
                return False
                
        time.sleep(poll_interval)
        
    print("[!] Auto-submitter timed out after", timeout_minutes, "minutes", flush=True)
    return False

if __name__ == '__main__':
    auto_submit_and_monitor()
