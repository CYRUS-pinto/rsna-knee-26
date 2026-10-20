"""
Watchdog for Submission 57338459 (Candidate V32-Ablation)
"""
import sys
import time
import subprocess
import json
from pathlib import Path

SUBMISSION_REF = "57338459"

print(f"Starting watcher for Submission Ref: {SUBMISSION_REF}...")

while True:
    res = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", "rsna-knee-abnormality-detection"],
        capture_output=True, text=True
    )
    if res.returncode == 0:
        lines = res.stdout.splitlines()
        for line in lines:
            if SUBMISSION_REF in line:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line.strip()}")
                if "SubmissionStatus.COMPLETE" in line:
                    print("EVALUATION COMPLETE!")
                    parts = line.split()
                    score = parts[-1] if len(parts) > 1 else "Unknown"
                    print(f"Final Public Score: {score}")
                    result = {
                        "ref": SUBMISSION_REF,
                        "status": "COMPLETE",
                        "public_score": score,
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    Path("submissions/v32_ablation_stride2_norad/output/lb_result.json").write_text(json.dumps(result, indent=2))
                    sys.exit(0)
                elif "SubmissionStatus.ERROR" in line:
                    print("EVALUATION FAILED WITH ERROR!")
                    sys.exit(1)
    time.sleep(60)
