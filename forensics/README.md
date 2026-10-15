# Forensics & Autopsy Archive

## Overview
Rigorous mathematical and empirical post-mortems documenting every regression, failed experiment, and competitor breakthrough.

## Documents
- **[`report-20261015-score-plateau.md`](report-20261015-score-plateau.md)**:
  - Deep forensic autopsy explaining why V7-V11 dropped to 0.955 while Leaderboard #1 reached 0.967.
  - Mathematical proof demonstrating why clinical conditional rules (`+0.15*ACL`) invert rankings on discordant patient subsets ($ACL^- / \text{Contusion}^+$ vs $ACL^+ / \text{Contusion}^-$).
  - Small-sample variance analysis proving why the 58 Gold studies overfit label noise (e.g., MCL has only 9 positives, Fracture only 7).
  - Compute credit ledger and A100 training runway.
