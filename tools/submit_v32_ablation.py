"""
Automated Submission Script for Candidate V32-Ablation
RSNA Knee Abnormality Detection Challenge
"""
import sys
import subprocess
from pathlib import Path

SUBMISSION_FILE = Path("submissions/v32_ablation_stride2_norad/output/submission.csv")
DESCRIPTION = "Candidate V32-Ablation (Raptor MultiScale + ConvNeXt R7 Stride-2 Twin, W_RAD=0)"

if not SUBMISSION_FILE.is_file():
    print(f"Error: Submission file {SUBMISSION_FILE} not found!", file=sys.stderr)
    sys.exit(1)

print(f"Verifying {SUBMISSION_FILE}...")
lines = SUBMISSION_FILE.read_text().splitlines()
print(f"Row count: {len(lines)}")
assert len(lines) == 4, f"Expected header + 3 rows, got {len(lines)}"

cmd = [
    "kaggle", "competitions", "submit",
    "-c", "rsna-knee-abnormality-detection",
    "-f", str(SUBMISSION_FILE),
    "-m", DESCRIPTION
]

print(f"Executing: {' '.join(cmd)}")
res = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)

if res.returncode == 0:
    print("SUCCESS: Submitted successfully to Kaggle!")
else:
    print(f"FAILED: Exit code {res.returncode}")
