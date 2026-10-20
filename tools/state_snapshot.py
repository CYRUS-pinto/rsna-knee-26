"""Refresh docs/LIVE_STATE.md with verified, timestamped competition facts.

Usage:  python tools/state_snapshot.py
Every agent MUST run this at session start and after every submission.
Never hand-edit LIVE_STATE.md; it is regenerated from the Kaggle API.
"""
from __future__ import annotations

import datetime as dt
import glob
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

COMP = "rsna-knee-abnormality-detection"
TEAM_NAME = "CYRUS PINTO"
DEADLINE = dt.datetime(2026, 10, 22, 23, 59, tzinfo=dt.timezone.utc)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "LIVE_STATE.md"
LB_DIR = ROOT / "scratch" / "lb_snapshot"


def submissions() -> list:
    import kaggle  # noqa: WPS433 (requires ~/.kaggle/kaggle.json or env vars)

    return list(kaggle.api.competition_submissions(COMP))


def leaderboard() -> pd.DataFrame | None:
    LB_DIR.mkdir(parents=True, exist_ok=True)
    for f in LB_DIR.glob("*.zip"):
        f.unlink()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    subprocess.run(
        ["kaggle", "competitions", "leaderboard", COMP, "--download", "-p", str(LB_DIR)],
        check=False, capture_output=True, env=env,
    )
    zips = sorted(glob.glob(str(LB_DIR / "*.zip")))
    if not zips:
        return None
    z = zipfile.ZipFile(zips[-1])  # filename contains ':' -> read in-memory (Windows-safe)
    df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    return df.sort_values("Score", ascending=False).reset_index(drop=True)


def main() -> int:
    now = dt.datetime.now(dt.timezone.utc)
    lines = [
        "# LIVE STATE (auto-generated — do not hand-edit)",
        "",
        f"- **Generated:** {now:%Y-%m-%d %H:%M} UTC by `tools/state_snapshot.py`",
        f"- **Deadline:** {DEADLINE:%Y-%m-%d %H:%M} UTC — **{(DEADLINE - now).total_seconds() / 3600:.1f} h remaining**",
        "",
    ]

    lb = leaderboard()
    if lb is not None:
        me = lb[lb.TeamName.str.upper() == TEAM_NAME]
        lines += ["## Leaderboard", ""]
        if len(me):
            r = me.iloc[0]
            lines.append(f"- **Our rank:** {int(r.Rank)} / {len(lb)} — score **{r.Score}** ({int(r.SubmissionCount)} subs)")
        lines.append(f"- **#1:** {lb.iloc[0].TeamName} — {lb.iloc[0].Score}")
        for k in (10, 15, 50, 100):
            if len(lb) >= k:
                lines.append(f"- **Rank {k} cutoff:** {lb.iloc[k - 1].Score}")
        for s in (0.969, 0.967, 0.966, 0.964, 0.962, 0.960, 0.958):
            lines.append(f"- Teams ≥ {s}: {int((lb.Score >= s).sum())}")
        lines.append("")

    try:
        subs = submissions()
    except Exception as exc:  # fail-soft: snapshot must never crash
        subs = []
        lines.append(f"> Submission fetch failed: {exc}")
    if subs:
        today = now.date()
        used = sum(1 for s in subs if s.date and s.date.date() == today)
        best = max((float(s.public_score) for s in subs if s.public_score), default=None)
        lines += [
            "## Submissions",
            "",
            f"- **Used today (UTC {today}):** {used} / 5 — **{5 - used} remaining** (team-wide quota)",
            f"- **Best public score:** {best}",
            "",
            "| Ref | UTC | Status | Public | Description |",
            "|---|---|---|---|---|",
        ]
        for s in subs[:25]:
            desc = (s.description or "").replace("|", "/")[:70]
            lines.append(f"| {s.ref} | {str(s.date)[:16]} | {str(s.status).split('.')[-1]} | {s.public_score or '—'} | {desc} |")
        lines.append("")

    lines += [
        "## Manual checks this script CANNOT do (verify in Kaggle UI)",
        "",
        "- Which 2 submissions are **selected** for final scoring (Submissions page → checkboxes).",
        "- **Hidden-test runtime** of each submission (needed for the Efficiency Prize).",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
