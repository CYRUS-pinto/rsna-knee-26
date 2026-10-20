---
trigger: always_on
---

# Always Document Every Step (RSNA Knee workspace)

Follow `AGENTS.md` at the workspace root. In short:

1. Session start: run `python tools/state_snapshot.py`, then read `docs/LIVE_STATE.md`, the tail of `docs/LIVE_STEP_LOG.md`, and `docs/RISK_REGISTER_TOP10.md`.
2. After every meaningful step (submit, push, train start/finish, score, error, decision, file change) run `python tools/log_step.py ...` with observed results and evidence.
3. Strategy changes also get an ADR in `decisions/`.
4. Never quote scores, ranks or quota from memory or old docs; quote `docs/LIVE_STATE.md`.
5. Session end: snapshot, final log entry with `--next`, commit docs.
