"""
Autonomous Overnight Submission Monitor for RSNA Knee Competition.
Continuously polls Candidate V48 (Ref 57443706) until COMPLETE.
Updates docs/LIVE_STATE.md, logs to docs/LIVE_STEP_LOG.md, and commits to git.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")
import kaggle

COMPETITION = "rsna-knee-abnormality-detection"
TARGET_REF = "57443706"
TARGET_DESC = "Candidate V48 Apex Quad-Arm Sovereign Ensemble (Engine 12 384px + Swin-B 384px + Engine 11 + Super-CoAtNet)"


def get_target_submission():
    try:
        subs = list(kaggle.api.competition_submissions(COMPETITION))
        for s in subs:
            if str(getattr(s, "ref", "")) == TARGET_REF:
                return s
    except Exception as e:
        print(f"[!] Error querying Kaggle API: {e}", flush=True)
    return None


def main():
    print(f"[*] Starting Autonomous Overnight Monitor for Ref {TARGET_REF}...", flush=True)
    kaggle.api.authenticate()

    poll_interval = 60
    t0 = time.time()

    while True:
        sub = get_target_submission()
        if sub is None:
            print(f"[!] Ref {TARGET_REF} not found. Retrying in {poll_interval}s...", flush=True)
            time.sleep(poll_interval)
            continue

        status = str(getattr(sub, "status", "")).upper()
        score = getattr(sub, "publicScore", None) or getattr(sub, "public_score", None)
        elapsed_min = int((time.time() - t0) / 60)

        print(f"[{elapsed_min}m elapsed] Ref {TARGET_REF} | Status: {status} | Score: {score}", flush=True)

        if "COMPLETE" in status:
            print(f"\n[+] SUCCESS: Ref {TARGET_REF} evaluation complete! Score: {score}", flush=True)

            # 1. Regenerate LIVE_STATE.md
            subprocess.run([sys.executable, "tools/state_snapshot.py"], check=False)

            # 2. Log step in LIVE_STEP_LOG.md
            result_msg = f"Public LB Score: {score}."
            if score and float(score) >= 0.959:
                result_msg += f" NEW BENCHMARK! Beat 0.958 PB. Eligible for Slot 2 selection."
                next_step = f"Lock V48 in Slot 2; prepare Candidate V49 with 5-fold calibrated weights."
            else:
                result_msg += f" Evaluated against 0.958 benchmark. Safety Slot 1 preserved."
                next_step = f"Train remaining folds 1-4 for Engine 12/13 and integrate DINOv2 features for V49."

            subprocess.run([
                sys.executable, "tools/log_step.py",
                "--agent", "Gemini 3.8 Flash",
                "--kind", "SCORE",
                "--action", f"{TARGET_DESC} Scored on Kaggle Leaderboard",
                "--result", result_msg,
                "--evidence", f"Kaggle Submission Ref {TARGET_REF}; score {score}",
                "--next", next_step
            ], check=False)

            # 3. Commit documentation to git
            subprocess.run(["git", "add", "docs/LIVE_STATE.md", "docs/LIVE_STEP_LOG.md"], check=False)
            subprocess.run(["git", "commit", "-m", f"docs: V48 scored {score} on Kaggle LB"], check=False)
            print("[+] State snapshot, step log, and git commit complete.", flush=True)
            break

        elif "ERROR" in status or "FAIL" in status:
            print(f"\n[!] Submission ERROR for Ref {TARGET_REF}: {status}", flush=True)
            subprocess.run([sys.executable, "tools/state_snapshot.py"], check=False)
            subprocess.run([
                sys.executable, "tools/log_step.py",
                "--agent", "Gemini 3.8 Flash",
                "--kind", "FAILURE",
                "--action", f"{TARGET_DESC} Failed on Kaggle Backend",
                "--result", f"Status: {status}. Kaggle worker encountered an error.",
                "--evidence", f"Kaggle Submission Ref {TARGET_REF}",
                "--next", "Inspect failure details and prepare clean fallback submission"
            ], check=False)
            break

        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
