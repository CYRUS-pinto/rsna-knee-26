"""
Watches Submission 57429824 (Candidate V47) until scoring completes,
then updates docs/LIVE_STATE.md and logs the score to docs/LIVE_STEP_LOG.md.
"""
import time, subprocess
from dotenv import load_dotenv

load_dotenv('.env')
import kaggle

SUB_REF = '57429824'
COMPETITION = 'rsna-knee-abnormality-detection'

def main():
    kaggle.api.authenticate()
    print(f"[*] Monitoring Submission {SUB_REF} (Candidate V47)...", flush=True)
    t0 = time.time()
    while time.time() - t0 < 3 * 3600:
        try:
            subs = kaggle.api.competition_submissions(COMPETITION)
            for s in subs:
                if str(getattr(s, 'ref', '')) == SUB_REF:
                    status = str(getattr(s, 'status', ''))
                    score = getattr(s, 'publicScore', None) or getattr(s, 'public_score', None)
                    elapsed = int(time.time() - t0)
                    print(f"[{elapsed}s] V47 Status: {status} | Score: {score}", flush=True)
                    
                    if 'COMPLETE' in status:
                        print(f"\\n[+] CANDIDATE V47 SCORED: {score}!", flush=True)
                        subprocess.run(['python', 'tools/state_snapshot.py'], check=False)
                        subprocess.run([
                            'python', 'tools/log_step.py',
                            '--agent', 'Gemini 3.8 Flash',
                            '--kind', 'SCORE',
                            '--action', 'Candidate V47 Scored on Kaggle Leaderboard',
                            '--result', f'Public LB Score: {score}',
                            '--evidence', f'Kaggle Submission Ref {SUB_REF}',
                            '--next', 'Analyze lift over 0.958 baseline and evaluate Swin-Base blend'
                        ], check=False)
                        return True
                    elif 'ERROR' in status:
                        print(f"\\n[!] V47 Failed scoring with status: {status}", flush=True)
                        return False
        except Exception as e:
            print(f"Poll error: {e}", flush=True)
            
        time.sleep(30)
        
    return False

if __name__ == '__main__':
    main()
