"""
Auto-submitter and Monitor for Candidate V46 Cross-View Rank Master.
Monitors cyruspinto/rsna-knee-v46-cross-view-rank-master until COMPLETE,
then submits to the RSNA Knee competition using today's quota (Submission 2/5 for Oct 20 UTC).
After submitting, monitors competition scoring status and updates docs/LIVE_STATE.md via tools/state_snapshot.py.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv(".env")
import kaggle


def auto_submit_and_monitor(
    kernel_slug="cyruspinto/rsna-knee-v46-cross-view-rank-master",
    poll_interval=30,
    timeout_minutes=180,
):
    kaggle.api.authenticate()
    print(
        f"[*] Starting Auto-Submitter for {kernel_slug} (Timeout: {timeout_minutes}m, Interval: {poll_interval}s)...",
        flush=True,
    )
    start_time = time.time()

    consecutive_errors = 0
    submitted = False
    sub_ref = None

    while time.time() - start_time < timeout_minutes * 60:
        try:
            if not submitted:
                status_res = kaggle.api.kernels_status(kernel_slug)
                curr_status = getattr(status_res, "status", None)
                elapsed = int(time.time() - start_time)
                print(f"[{elapsed}s] Kernel Status: {curr_status}", flush=True)
                consecutive_errors = 0

                if "COMPLETE" in str(curr_status):
                    print(
                        f"[+] Kernel {kernel_slug} is COMPLETE! Submitting to Kaggle competition...",
                        flush=True,
                    )
                    res = kaggle.api.competition_submit_code(
                        file_name="submission.csv",
                        message="Candidate V46 Cross-View Rank Master (Multi-Planar Spatial Decoupling)",
                        competition="rsna-knee-abnormality-detection",
                        kernel=kernel_slug,
                        kernel_version=1,
                    )
                    print(
                        "[+] SUBMISSION SUCCESSFUL! Kaggle Response:",
                        res,
                        flush=True,
                    )
                    submitted = True
                    time.sleep(10)
                    subs = kaggle.api.competition_submissions(
                        "rsna-knee-abnormality-detection"
                    )
                    if subs:
                        sub_ref = str(subs[0].ref)
                        print(
                            f"[*] Tracked new competition submission Ref: {sub_ref}",
                            flush=True,
                        )
                elif "ERROR" in str(curr_status):
                    failure_msg = getattr(
                        status_res, "failure_message", "No details"
                    )
                    print(
                        f"[!] Kernel {kernel_slug} failed with ERROR: {failure_msg}",
                        flush=True,
                    )
                    return False
            else:
                elapsed = int(time.time() - start_time)
                subs = kaggle.api.competition_submissions(
                    "rsna-knee-abnormality-detection"
                )
                target_sub = (
                    next((s for s in subs if str(s.ref) == sub_ref), None)
                    if sub_ref
                    else subs[0]
                )

                sub_status = getattr(target_sub, "status", None)
                score = getattr(target_sub, "public_score", None)
                print(
                    f"[{elapsed}s] Submission {sub_ref} Status: {sub_status} | Public Score: {score}",
                    flush=True,
                )

                if "COMPLETE" in str(sub_status):
                    print(
                        f"🚀 [SUCCESS] Submission {sub_ref} EVALUATION COMPLETE! Public LB Score: {score}",
                        flush=True,
                    )
                    # Trigger snapshot update
                    os.system("python tools/state_snapshot.py")
                    os.system(
                        f'python tools/log_step.py --agent "Auto-Daemon" --kind SCORE --action "Candidate V46 Scored" --result "Public score: {score}" --evidence "kaggle API ref {sub_ref}" --next "Evaluate V46 vs 0.958 PB"'
                    )
                    return True
                elif "ERROR" in str(sub_status):
                    print(
                        f"[!] Submission {sub_ref} FAILED during scoring!",
                        flush=True,
                    )
                    return False

        except Exception as e:
            consecutive_errors += 1
            print(f"[!] Warning: Exception during polling ({e})", flush=True)
            if consecutive_errors > 5:
                time.sleep(60)

        time.sleep(poll_interval)

    print("[!] Auto-submitter timed out.", flush=True)
    return False


if __name__ == "__main__":
    auto_submit_and_monitor()
