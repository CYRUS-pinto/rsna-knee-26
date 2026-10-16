import os
import sys
import time
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

LOG_FILE = Path('docs/competition_live_log.md')

def log_event(message):
    ts = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    entry = f"- `[{ts}]` {message}\n"
    print(entry.strip())
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(entry)

def check_v17_submission(v17_ref='57253840'):
    try:
        kaggle.api.authenticate()
        subs = kaggle.api.competition_submissions('rsna-knee-abnormality-detection')
        for s in subs:
            if str(getattr(s, 'ref', '')) == str(v17_ref):
                status = str(getattr(s, 'status', ''))
                score = getattr(s, 'public_score', None)
                return status, score
    except Exception as e:
        return 'ERROR', str(e)
    return 'NOT_FOUND', None

def check_v18_kernel(kernel_id='cyruspinto/rsna-knee-v18-grandmaster-blend'):
    try:
        kaggle.api.authenticate()
        status_obj = kaggle.api.kernels_status(kernel_id)
        status = getattr(status_obj, 'status', str(status_obj))
        return str(status)
    except Exception as e:
        return f'ERROR: {e}'

def run_daemon(poll_interval=90, max_hours=6):
    max_loops = int((max_hours * 3600) / poll_interval)
    log_event("Competition Monitor Daemon started (polling every 90s).")
    
    v17_scored = False
    v18_submitted = False

    for loop in range(max_loops):
        # 1. Check V17 Submission
        if not v17_scored:
            status, score = check_v17_submission('57253840')
            if 'COMPLETE' in status and score is not None:
                log_event(f"**V17 SUBMISSION COMPLETE!** Public Score: `{score}` (Ref: `57253840`)")
                v17_scored = True
            elif 'PENDING' in status:
                print(f"[{time.strftime('%X')}] V17 is still scoring on Kaggle...")
            elif 'ERROR' in status:
                print(f"[{time.strftime('%X')}] V17 status query retry: {score}")

        # 2. Check V18 Kernel
        if not v18_submitted:
            k_status = check_v18_kernel('cyruspinto/rsna-knee-v18-grandmaster-blend')
            print(f"[{time.strftime('%X')}] V18 Kernel Status: {k_status}")
            
            if 'COMPLETE' in k_status:
                log_event("V18 Kernel completed execution on Kaggle Dual T4 GPUs! Downloading output...")
                out_dir = Path('submissions/v18_grandmaster_calibrated/output')
                out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    kaggle.api.kernels_output('cyruspinto/rsna-knee-v18-grandmaster-blend', path=str(out_dir))
                    log_event(f"Downloaded V18 artifacts to {out_dir}")
                    
                    sub_file = out_dir / 'submission.csv'
                    if sub_file.exists():
                        # Verify integrity
                        res = subprocess.run([sys.executable, 'tools/verify_submission_integrity.py', str(sub_file)], capture_output=True, text=True)
                        if res.returncode == 0:
                            log_event("V18 submission.csv passed integrity audit! Submitting to competition...")
                            sub_res = kaggle.api.competition_submit(
                                str(sub_file),
                                "V18 Grandmaster MultiScale + ConvNeXt R6 Flip TTA + Asymmetric Weights + Soft Synovitis Floor",
                                "rsna-knee-abnormality-detection"
                            )
                            log_event(f"**V18 OFFICIALLY SUBMITTED!** Message: {sub_res}")
                            v18_submitted = True
                        else:
                            log_event(f"V18 integrity check failed: {res.stdout}")
                except Exception as e:
                    log_event(f"Error handling V18 completion: {e}")
            elif 'ERROR' in k_status:
                pass

        if v17_scored and v18_submitted:
            log_event("All active candidate pipelines (V17 and V18) successfully processed and logged!")
            break

        time.sleep(poll_interval)

if __name__ == '__main__':
    run_daemon()
