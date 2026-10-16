import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('.env')

import kaggle

def attempt_push_v18():
    folder = str(Path('submissions/v18_grandmaster_calibrated').resolve())
    try:
        kaggle.api.authenticate()
        print(f"[{time.strftime('%X')}] Attempting to push Candidate V18 to Kaggle...")
        res = kaggle.api.kernels_push(folder)
        error = getattr(res, 'error', None)
        if error and 'Maximum batch GPU session count' in str(error):
            print(f"[{time.strftime('%X')}] Still at 2/2 GPU capacity ({error}). Standing by...")
            return False
        elif error:
            print(f"[{time.strftime('%X')}] Push returned error: {error}")
            return False
        else:
            print(f"[{time.strftime('%X')}] SUCCESS! Candidate V18 pushed and launched on Kaggle!")
            print("Response:", res)
            return True
    except Exception as e:
        print(f"[{time.strftime('%X')}] Connection / Introspect error: {type(e).__name__}. Will retry...")
        return False

def monitor_and_launch(interval_seconds=60, max_hours=3):
    max_loops = int((max_hours * 3600) / interval_seconds)
    print(f"[{time.strftime('%X')}] Starting Auto-Launch Daemon for Candidate V18 (interval: {interval_seconds}s)")
    for i in range(max_loops):
        success = attempt_push_v18()
        if success:
            print(f"[{time.strftime('%X')}] Candidate V18 is now actively running on Kaggle GPUs!")
            break
        time.sleep(interval_seconds)

if __name__ == '__main__':
    monitor_and_launch()
