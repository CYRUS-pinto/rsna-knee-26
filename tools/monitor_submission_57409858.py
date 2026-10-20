"""
Monitor script for Candidate V44 (Submission Ref: 57409858).
Polls every 30s until COMPLETE or ERROR.
"""
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv('.env')
import kaggle

def monitor_submission(sub_id='57409858', poll_interval=30):
    kaggle.api.authenticate()
    print(f"[*] Monitoring Competition Submission {sub_id} (Candidate V44 Sovereign Pinnacle Master)...", flush=True)
    start_time = time.time()
    
    while True:
        try:
            subs = kaggle.api.competition_submissions('rsna-knee-abnormality-detection')
            target_sub = None
            for s in subs:
                if str(s.ref) == str(sub_id):
                    target_sub = s
                    break
            
            elapsed = int(time.time() - start_time)
            if target_sub:
                status = getattr(target_sub, 'status', 'UNKNOWN')
                score = getattr(target_sub, 'public_score', None)
                print(f"[{elapsed}s] Sub {sub_id} Status: {status} | Score: {score}", flush=True)
                
                if 'COMPLETE' in str(status) and score not in [None, '']:
                    print(f"==================================================")
                    print(f"[+] SUBMISSION {sub_id} SCORING COMPLETE!")
                    print(f"[+] PUBLIC LEADERBOARD SCORE: {score}")
                    print(f"==================================================")
                    break
                elif 'ERROR' in str(status):
                    print(f"[!] SUBMISSION {sub_id} FAILED WITH ERROR.")
                    break
            else:
                print(f"[{elapsed}s] Sub {sub_id} not found in top submissions.", flush=True)
                
        except Exception as e:
            print("Poll error:", type(e).__name__, str(e)[:100], flush=True)
            
        time.sleep(poll_interval)

if __name__ == '__main__':
    monitor_submission()
