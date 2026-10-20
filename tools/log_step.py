"""Append one entry to docs/LIVE_STEP_LOG.md (append-only, never rewrite history).

Usage:
  python tools/log_step.py --agent "Claude Opus" --action "Submitted V46" \
      --result "ref 57xxxxx PENDING" --evidence "kaggle competitions submissions" \
      --next "poll score in 60 min"
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "docs" / "LIVE_STEP_LOG.md"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, help="model / human name")
    p.add_argument("--action", required=True)
    p.add_argument("--result", required=True, help="observed outcome, not intent")
    p.add_argument("--evidence", default="", help="command, file, ref id, or URL")
    p.add_argument("--next", default="", help="single next step")
    p.add_argument("--kind", default="STEP", choices=["STEP", "DECISION", "FAILURE", "SUBMIT", "SCORE", "TRAIN"])
    a = p.parse_args()

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = (
        f"\n### [{ts}] {a.kind} — {a.action}\n"
        f"- **Agent:** {a.agent}\n"
        f"- **Result:** {a.result}\n"
        + (f"- **Evidence:** {a.evidence}\n" if a.evidence else "")
        + (f"- **Next:** {a.next}\n" if a.next else "")
    )
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(entry)
    print(entry)


if __name__ == "__main__":
    main()
