# 10 Long-Run Assets — What Makes Every Future Session (and Competition) Faster

**Created:** 2026-10-20 · Complements (does not replace) the 10 assets in [ADR-043](../decisions/ADR-043-grandmaster-operating-protocol-and-ten-long-run-assets.md), which cover the guardrail, logit sanitizer, metric module, OOF stacking, Colab pipeline, DICOM alignment, dual-branch conditioning, two-track doctrine and ADR ledger.

| # | Asset | Status | Where | Why it pays off long-run |
|---|---|---|---|---|
| 1 | **Auto-generated live state** | ✅ Built & tested | `tools/state_snapshot.py` → `docs/LIVE_STATE.md` | Rank, quota, scores, LB cutoffs, hours left — from the API, never from memory. Killed the "stale numbers" failure mode (E1). |
| 2 | **Append-only step log** | ✅ Built | `tools/log_step.py` → `docs/LIVE_STEP_LOG.md` | Any agent (Gemini, Claude, GPT, human) resumes from the last `--next`. Uniform format = greppable history. |
| 3 | **Universal agent entry point** | ✅ Built | `AGENTS.md` + `.agents/rules/always-document.md` | Auto-loaded by most harnesses; enforces read order and logging after every step. |
| 4 | **Exhaustive risk register** | ✅ Built | `docs/RISK_REGISTER_TOP10.md` | 60+ failure modes with detection + prevention; ✅ marks ones that already cost us. |
| 5 | **Training-script preflight** | ✅ Built & run on E12/E13 (PASS) | `tools/preflight_train_script.py` | Catches syntax errors, missing grad-checkpointing, Drive hangs, end-only checkpoints *before* Colab burns units. |
| 6 | **Hypothesis-first submission template** | 📄 Convention | Log entry `--kind SUBMIT` must contain: hypothesis, single change vs parent, expected Δ, OOF evidence | Turns each of the ~13 remaining submissions into a controlled experiment instead of a guess (A5). |
| 7 | **Weights manifest** (planned) | ⏳ To build when E12/E13 land | `weights/MANIFEST.csv`: slug, file, sha256, arch, img size, norm, fold, val AUC, timm/torch versions | Prevents wrong-checkpoint/normalization/version mismatches (B13, B14, C9, C12, D5). |
| 8 | **OOF prediction store** (planned) | ⏳ | `results/oof/<engine>_fold<k>.parquet` with identical columns | Every new arm becomes blendable and weight-optimizable in minutes via `optimize_pathology_stacking_oof.py`; correlation check for pseudo-diversity (A2). |
| 9 | **Final-selection checklist** | 📄 In Encyclopedia §9 | T-24 h and T-6 h manual Kaggle-UI check, logged | Prevents the single most expensive mistake: wrong final pair (D12, D13). |
| 10 | **Cross-competition lessons file** | ✅ Built | `docs/LESSONS_FOR_FUTURE_COMPETITIONS.md` | Portable lessons for the next Kaggle medical-imaging competition. |
