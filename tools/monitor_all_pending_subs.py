"""
Unified Live Submission Monitor for RSNA Knee Competition.
Tracks all PENDING submissions (Candidate V46: 57426806, Candidate V47: 57429824)
until scoring completes, then updates docs/LIVE_STATE.md and logs the scores.
"""
import time, subprocess
from dotenv import load_dotenv

load_dotenv('.env')
import kaggle

COMPETITION = 'rsna-knee-abnormality-detection'
TRACK_REFS = {'57426806': 'Candidate V46 Cross-View Rank Master', '57429824': 'Candidate V47 Apex Tri-Arm Ensemble'}

def main():
    kaggle.api.authenticate()
    print(f"[*] Starting Unified Score Monitor for: {TRACK_REFS} (Timeout: 6 hours)...", flush=True)
    t0 = time.time()
    completed = set()

    while time.time() - t0 < 6 * 3600 and len(completed) < len(TRACK_REFS):
        try:
            subs = list(kaggle.api.competition_submissions(COMPETITION))
            for s in subs:
                ref_str = str(getattr(s, 'ref', ''))
                if ref_str in TRACK_REFS and ref_str not in completed:
                    status = str(getattr(s, 'status', ''))
                    score = getattr(s, 'publicScore', None) or getattr(s, 'public_score', None)
                    elapsed = int(time.time() - t0)
                    print(f"[{elapsed}s] Ref {ref_str} ({TRACK_REFS[ref_str]}) Status: {status} | Score: {score}", flush=True)

                    if 'COMPLETE' in status:
                        print(f"\n[+] REF {ref_str} SCORED: {score}!", flush=True)
                        completed.add(ref_str)
                        subprocess.run(['python', 'tools/state_snapshot.py'], check=False)
                        subprocess.run([
                            'python', 'tools/log_step.py',
                            '--agent', 'Gemini 3.8 Flash',
                            '--kind', 'SCORE',
                            '--action', f'{TRACK_REFS[ref_str]} Scored on Kaggle Leaderboard',
                            '--result', f'Public LB Score: {score}',
                            '--evidence', f'Kaggle Submission Ref {ref_str}',
                            '--next', 'Evaluate against 0.958 baseline and proceed with multi-model blend'
                        ], check=False)
                    elif 'ERROR' in status:
                        print(f"\n[!] Ref {ref_str} Error with status: {status}", flush=True)
                        completed.add(ref_str)
        except Exception as e:
            print(f"Poll error: {e}", flush=True)

        time.sleep(30)

    print(f"[+] All monitored submissions processed! Elapsed: {int(time.time() - t0)}s", flush=True)

if __name__ == '__main__':
    main()
