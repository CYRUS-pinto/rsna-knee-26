# AGENTS.md — Mandatory Operating Protocol (read FIRST, every session, every model)

This file is read automatically by Antigravity/Gemini, Claude Code, Codex, Cursor and others.
The user switches AI agents often. **Continuity depends entirely on the files below being current.**

## 1. Session start (in this order, before any other work)

1. `python tools/state_snapshot.py` → regenerates [docs/LIVE_STATE.md](docs/LIVE_STATE.md) from the Kaggle API (rank, quota, scores, hours left).
2. Read [docs/LIVE_STATE.md](docs/LIVE_STATE.md) — the **only** source for numbers. If any other doc disagrees, LIVE_STATE wins.
3. Read the last 30 entries of [docs/LIVE_STEP_LOG.md](docs/LIVE_STEP_LOG.md) — what the previous agent did and what it planned next.
4. Read [docs/RISK_REGISTER_TOP10.md](docs/RISK_REGISTER_TOP10.md) — everything that can go wrong.
5. Read [docs/RSNA_KNEE_2026_MASTER_ENCYCLOPEDIA_AND_HANDOFF.md](docs/RSNA_KNEE_2026_MASTER_ENCYCLOPEDIA_AND_HANDOFF.md) — history, architecture, rules.
6. Only then act.

## 2. Logging rule — after EVERY meaningful step (no exceptions)

```
python tools/log_step.py --agent "<model name>" --kind <STEP|DECISION|FAILURE|SUBMIT|SCORE|TRAIN> \
  --action "<what you did>" --result "<what actually happened>" \
  --evidence "<command / file / ref id>" --next "<single next step>"
```

- "Meaningful" = any submission, kernel push, dataset push, training start/finish, score received, error hit, decision taken, file created/changed.
- Log **observed results**, never intentions. "Submitted, ref 574xxxxx PENDING" — not "will submit".
- A FAILURE entry must include the exact error line and the root cause (or "root cause unknown").
- A DECISION that changes strategy also gets an ADR in `decisions/` (next free number; currently ADR-047+).
- After any submission or score: re-run `python tools/state_snapshot.py`.

## 3. Hard rules (violating any of these has cost us real score before)

1. **Never deselect the 0.958 safety submission** unless a candidate scores ≥ 0.9610 public. Verify selections in the Kaggle UI before 2026-10-22 23:59 UTC.
2. **Rank-percentile blending only.** No raw-probability averaging across model families, no power sharpening (p>1). (V42 → 0.943.)
3. **No hand-tuned clinical "boost" priors** in submissions unless proven on full OOF (ADR-002: V12 dropped to 0.953).
4. **No fatal asserts / raises on test data.** Run `python tools/validate_kernel_guardrail.py <notebook>` before every push. (V34–V36 crashed.)
5. **New model weight ≤ 15% of the blend** until OOF-optimized weights exist. (V42.)
6. **Gold-58 is a sanity check, never a tuning target** (7 fractures, 9 MCL tears).
7. **Training = isolated script** (`!python train_x.py`), `set_grad_checkpointing(True)`, `chunk_size ≤ 4` for 384px backbones.
8. **No credentials in files, notebooks, logs or prompts.** Use env vars / Colab Secrets.
9. **State only what you verified.** If you did not run it, say "unverified".
10. **Quota is team-wide (5 members) and resets 00:00 UTC (05:30 IST).** Check LIVE_STATE before submitting.

## 4. Session end

1. `python tools/state_snapshot.py`
2. Log a final STEP entry with `--next` describing exactly where to resume.
3. `git add docs decisions tools AGENTS.md && git commit -m "docs: session handoff <UTC time>"` (never commit `.env`, weights, or data).
