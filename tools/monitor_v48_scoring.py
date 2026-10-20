"""
Unified Live Submission Monitor for RSNA Knee Competition.
Monitors V48 kernel execution until COMPLETE, then calls competition_submit_code
to submit to the Kaggle competition, and polls until scoring completes.
Logs scores and updates docs/LIVE_STATE.md automatically.
"""
import os
import sys
import time
import subprocess
from dotenv import load_dotenv

load_dotenv('.env')
import kaggle

COMPETITION = 'rsna-knee-abnormality-detection'
KERNEL_SLUG = 'cyruspinto/rsna-knee-v48-apex-quad-arm-ensemble'
LABEL = 'Candidate V48 Apex Quad-Arm Sovereign Ensemble (Engine 12 384px + Swin-B 384px + Engine 11 + Super-CoAtNet)'

def poll_kernel_until_complete(timeout=3*3600):
    """Wait for V48 kernel to finish running (QUEUED -> RUNNING -> COMPLETE/ERROR)."""
    t0 = time.time()
    last_status = None
    print(f"[*] Polling kernel {KERNEL_SLUG} for completion (timeout {timeout//3600}h)...", flush=True)
    while time.time() - t0 < timeout:
        try:
            resp = kaggle.api.kernels_status(KERNEL_SLUG)
            status = str(getattr(resp, 'status', '')).upper()
            if status != last_status:
                print(f"  [kernel] Status -> {status}", flush=True)
                last_status = status
            if 'COMPLETE' in status:
                print(f"[+] Kernel COMPLETE!", flush=True)
                return True
            if 'ERROR' in status or 'CANCEL' in status or 'FAIL' in status:
                fail_msg = getattr(resp, 'failure_message', None) or getattr(resp, 'failureMessage', None)
                print(f"[!] Kernel ERROR: {status} | Details: {fail_msg}", flush=True)
                return False
        except Exception as e:
            print(f"  kernel poll error: {e}", flush=True)
        time.sleep(30)
    print("[!] Kernel poll timeout.", flush=True)
    return False

def submit_kernel():
    """Submit the completed kernel to the competition."""
    print(f"[*] Submitting kernel {KERNEL_SLUG} to {COMPETITION}...", flush=True)
    try:
        res = kaggle.api.competition_submit_code(
            file_name='submission.csv',
            message=LABEL,
            competition=COMPETITION,
            kernel=KERNEL_SLUG
        )
        print(f"[+] Competition submit successful: {res}", flush=True)
        ref = getattr(res, 'ref', None)
        if ref:
            return str(ref)
    except Exception as e:
        print(f"[!] Error calling competition_submit_code: {e}", flush=True)
    
    # Fallback to inspecting newest submission
    time.sleep(10)
    try:
        subs = list(kaggle.api.competition_submissions(COMPETITION))
        if subs:
            return str(getattr(subs[0], 'ref', ''))
    except Exception as e:
        print(f"[!] Error inspecting submissions: {e}", flush=True)
    return None

def poll_submission_score(ref_str, timeout=2*3600):
    """Poll competition submissions until ref_str is COMPLETE."""
    t0 = time.time()
    print(f"[*] Polling submission ref {ref_str} for score (timeout {timeout//3600}h)...", flush=True)
    while time.time() - t0 < timeout:
        try:
            subs = list(kaggle.api.competition_submissions(COMPETITION))
            for s in subs:
                if str(getattr(s, 'ref', '')) == ref_str:
                    status = str(getattr(s, 'status', ''))
                    score = getattr(s, 'publicScore', None) or getattr(s, 'public_score', None)
                    elapsed = int(time.time() - t0)
                    print(f"[{elapsed}s] Ref {ref_str} Status: {status} | Score: {score}", flush=True)
                    if 'COMPLETE' in status:
                        print(f"\n[+] REF {ref_str} SCORED: {score}!", flush=True)
                        return score
                    if 'ERROR' in status:
                        print(f"\n[!] Submission ERROR: {status}", flush=True)
                        return None
        except Exception as e:
            print(f"  poll error: {e}", flush=True)
        time.sleep(30)
    return None

def main():
    kaggle.api.authenticate()
    print(f"[*] Starting V48 Monitor | {LABEL}", flush=True)

    # Step 1: Wait for kernel to finish
    kernel_ok = poll_kernel_until_complete(timeout=3*3600)
    if not kernel_ok:
        print("[!] Kernel did not complete successfully. Exiting.", flush=True)
        return

    # Step 2: Submit to Kaggle competition
    v48_ref = submit_kernel()
    if not v48_ref:
        print("[!] Could not obtain V48 submission ref. Exiting.", flush=True)
        return

    print(f"[+] V48 submission ref tracked: {v48_ref}", flush=True)

    # Step 3: Poll for score
    score = poll_submission_score(v48_ref, timeout=2*3600)

    # Step 4: Update state and log
    subprocess.run(['python', 'tools/state_snapshot.py'], check=False)
    subprocess.run([
        'python', 'tools/log_step.py',
        '--agent', 'Gemini 3.8 Flash',
        '--kind', 'SCORE',
        '--action', f'{LABEL} Scored on Kaggle Leaderboard',
        '--result', f'Public LB Score: {score}',
        '--evidence', f'Kaggle Submission Ref {v48_ref}',
        '--next', 'If score >= 0.959 select in Slot 2; else analyze delta and plan V49 with OOF weight calibration'
    ], check=False)

    # Step 5: Commit docs
    subprocess.run([
        'git', 'add', 'docs/LIVE_STATE.md', 'docs/LIVE_STEP_LOG.md'
    ], check=False)
    subprocess.run([
        'git', 'commit', '-m', f'docs: V48 scored {score} on Kaggle LB'
    ], check=False)

    print(f"\n[+] All monitored submissions processed! Score: {score}", flush=True)

if __name__ == '__main__':
    main()
