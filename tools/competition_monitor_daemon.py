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

def check_submission(sub_ref):
    try:
        kaggle.api.authenticate()
        subs = kaggle.api.competition_submissions('rsna-knee-abnormality-detection')
        for s in subs:
            if str(getattr(s, 'ref', '')) == str(sub_ref):
                status = str(getattr(s, 'status', ''))
                score = getattr(s, 'public_score', None)
                error_desc = getattr(s, 'error_description', None)
                return status, score, error_desc
    except Exception as e:
        return 'ERROR', None, str(e)
    return 'NOT_FOUND', None, None

def run_daemon(poll_interval=120, max_hours=8):
    max_loops = int((max_hours * 3600) / poll_interval)
    log_event("Competition Monitor Daemon started (tracking V17 [57253840] and V18 [57260494]).")
    
    v17_done = False
    v18_done = False

    for loop in range(max_loops):
        # 1. Check V17 (57253840)
        if not v17_done:
            status, score, err = check_submission('57253840')
            if 'COMPLETE' in status and score is not None:
                log_event(f"**V17 SUBMISSION COMPLETE!** Public Score: `{score}` (Ref: `57253840`)")
                v17_done = True
            elif 'ERROR' in status:
                log_event(f"V17 Status Error: {err}")
            else:
                print(f"[{time.strftime('%X')}] V17 (57253840) status: {status}")

        # 2. Check V18 (57260494)
        if not v18_done:
            status, score, err = check_submission('57260494')
            if 'COMPLETE' in status and score is not None:
                log_event(f"**V18 SUBMISSION COMPLETE!** Public Score: `{score}` (Ref: `57260494`)")
                v18_done = True
            elif 'ERROR' in status:
                log_event(f"V18 Status Error: {err}")
            else:
                print(f"[{time.strftime('%X')}] V18 (57260494) status: {status}")

        if v17_done and v18_done:
            log_event("Both V17 and V18 submissions have completed evaluation!")
            break

        time.sleep(poll_interval)

if __name__ == '__main__':
    run_daemon()
