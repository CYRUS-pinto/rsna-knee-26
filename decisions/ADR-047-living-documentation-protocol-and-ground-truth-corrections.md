# ADR-047: Living Documentation Protocol, Ground-Truth Corrections, and the Real Gap to Top 10

- **Status:** ACCEPTED
- **Date:** 2026-10-20
- **Supersedes (in part):** ADR-043 §1 (stale rank and efficiency formula), the V47 "kinematic coupling" plan in the 2026-10-20 Encyclopedia (it conflicts with ADR-002)

## Context
The user switches AI agents frequently. Handoffs were hand-written, went stale within hours, and spread wrong numbers between agents:
- the docs said rank 583, top-10 at 0.966, and V45 pending; in reality it was rank 688, top-10 at 0.967, and V45 had scored 0.958
- the efficiency formula in the guardrail skill was wrong
- a 3-study interactive runtime was presented as the efficiency-prize runtime.

## Verified facts (2026-10-20 06:30 UTC)
- Rank 688 / 6,255, best 0.958, 34 submissions, team of 5.
- 1,542 teams at ≥ 0.958 (public-notebook plateau). Score cutoffs: rank 10 = 0.967, rank 50 = 0.963, rank 100 = 0.960. #1 = 0.969.
- Deadline 2026-10-22 23:59 UTC (Kaggle CLI).

## Decision
1. Ground truth comes from `tools/state_snapshot.py` → `docs/LIVE_STATE.md` only.
2. Every meaningful step is logged with `tools/log_step.py` → `docs/LIVE_STEP_LOG.md`.
3. `AGENTS.md` (root) and `.agents/rules/always-document.md` enforce the read order and the logging rule.
4. `docs/RISK_REGISTER_TOP10.md` is the canonical failure-mode list.
5. Training scripts must pass `tools/preflight_train_script.py` before they are handed to Colab.
6. The V47 blend uses OOF-fitted rank weights (new arms capped at 15% without OOF) and **no clinical boosts**.
7. The max-AUC final candidate is **not** capped at 500 s. The efficiency prize is a separate candidate.

## Consequences
- Any agent can resume with zero chat history.
- Strategy now focuses on new signal (new engines, more folds, OOF stacking) instead of re-weighting arms we already have, because re-weighting cannot leave the 0.958 plateau.
