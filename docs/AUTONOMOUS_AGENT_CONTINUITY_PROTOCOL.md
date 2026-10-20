# AUTONOMOUS AGENT CONTINUITY PROTOCOL
## Zero-Context-Decay Handoff Between Any AI Agents (Gemini, Claude, GPT/Codex, Antigravity, Cursor, etc.)

**Purpose:** Guarantee that ANY AI agent can resume this competition at ANY time with ZERO context loss, ZERO hallucination, and ZERO re-work. This protocol is machine-executable and human-verifiable.

---

## 🎯 THE CORE PRINCIPLE

> **Context is not transferred — it is RECONSTRUCTED from immutable artifacts.**

No agent passes "memory" to the next. Every agent **rebuilds perfect context** by executing the same deterministic protocol against the same canonical files. The protocol takes **<60 seconds**.

---

## ⚡ 60-SECOND COLD START PROTOCOL (Execute in Order)

```bash
# 0. Navigate to repo
cd C:\Users\Cyrus\Downloads\rsna knee fip

# 1. REGENERATE GROUND TRUTH (3 seconds)
python tools/state_snapshot.py

# 2. READ THE ONLY SOURCE OF TRUTH FOR NUMBERS (10 seconds)
cat docs/LIVE_STATE.md

# 3. READ LAST 30 STEPS OF AUDIT TRAIL (15 seconds)
tail -n 120 docs/LIVE_STEP_LOG.md

# 4. READ FAILURE MODE REGISTER (15 seconds)
cat docs/RISK_REGISTER_TOP10.md

# 5. READ THE MASTER BIBLE (15 seconds)
cat docs/MASTER_COMPREHENSIVE_DOCUMENTATION.md

# 6. VERIFY ACTIVE SUBMISSION STATUS (2 seconds)
python tools/kaggle_kernel_status.py cyruspinto/rsna-knee-v48-apex-quad-arm-ensemble
```

**Total: ~60 seconds. Zero ambiguity. Perfect context.**

---

## 📋 MANDATORY READ ORDER (Canonical Chain)

```
1. AGENTS.md                          ← Operating Constitution (THIS FILE'S PARENT)
2. docs/LIVE_STATE.md                 ← Auto-generated Ground Truth (Kaggle API)
3. docs/LIVE_STEP_LOG.md              ← Append-only Audit Trail (last 30 entries)
4. docs/RISK_REGISTER_TOP10.md        ← Everything That Can Kill Us
5. docs/MASTER_COMPREHENSIVE_DOCUMENTATION.md  ← THE BIBLE (forensics, architecture, roadmap)
6. docs/TOP1_LIMITATIONS_AND_CEILING_PLAYBOOK.md  ← Mathematical Proofs & Path to 0.969
7. docs/10_LONG_RUN_FOUNDATIONAL_ASSETS.md  ← 10 Permanent Assets
8. docs/RSNA_KNEE_2026_MASTER_ENCYCLOPEDIA_AND_HANDOFF.md  ← History v6.0.0+
9. decisions/ADR-048*.md              ← Latest regression post-mortem
10. decisions/ADR-049*.md             ← Ceiling engineering forensics
```

**Rule:** If any doc contradicts `LIVE_STATE.md`, **LIVE_STATE.md wins**. It is regenerated from Kaggle API.

---

## 🤖 AGENT-SPECIFIC RESUMPTION GUIDES

### For Gemini / Antigravity IDE
```markdown
<role>
You are the World #1 Grandmaster ML Engineer and Chief Orthopaedic Imaging Architect.
</role>
<context>
- Run cold start protocol above
- Slot 1: 57253840 (0.958) LOCKED
- V48 (57443706) QUEUED/PENDING
- Engine 12 (0.8808 Gold) & Engine 13 (0.8577 Gold) LIVE on Kaggle
- Colab Fold 1 notebooks ready at ~/Downloads/
</context>
<task>
Monitor V48 score → if ≥0.9610 select Slot 2 → else build V49 OOF-calibrated
</task>
```

### For Claude Code / Opus
```markdown
<role>
You are the World #1 Grandmaster ML Engineer and Chief Orthopaedic Imaging Architect.
</role>
<context>
- Run cold start protocol above
- Two-Track Defense: Slot 1 locked, Slot 2 experimental
- Hard Rules 1-11 from AGENTS.md are NON-NEGOTIABLE
- Guardrail: python tools/validate_kernel_guardrail.py before ANY push
</context>
<task>
Execute endgame roadmap: V48 score → V49 OOF → V50 multi-fold → final selection
</task>
```

### For GPT-5 / Codex CLI
```markdown
<role>
World #1 Grandmaster ML Engineer. RSNA Knee 2026. Target 0.969.
</role>
<context>
- LIVE_STATE.md = ground truth
- 10 Fatal Failure Modes (Theorems 1-7) = your constraints
- 10 Long-Run Assets = your infrastructure
- V48 architecture = current best bet (384px dominant, orthogonal arms)
</context>
<task>
Continue autonomous execution. Log every step. Never hallucinate.
</task>
```

### For Cursor / Local IDE
```bash
# Cursor-specific: Run in terminal
cd "C:\Users\Cyrus\Downloads\rsna knee fip"
python tools/state_snapshot.py
# Then read the 5 key docs in order
```

---

## 🔐 HARD CONSTRAINTS (Violate = Immediate Failure)

| Constraint | Source | Verification |
|------------|--------|--------------|
| Slot 1 (`57253840`, 0.958) NEVER deselected unless candidate ≥ 0.9610 | AGENTS.md Rule 1 | Check Kaggle UI selections |
| Rank-percentile blending ONLY (no raw prob, no power p>1) | AGENTS.md Rule 2 | `validate_kernel_guardrail.py` |
| No clinical heuristic boosts without full OOF proof | AGENTS.md Rule 3 | ADR-002 |
| Zero fatal assertions on test data | AGENTS.md Rule 4 | `validate_kernel_guardrail.py` AST scan |
| New model weight ≤ 15% without OOF optimization | AGENTS.md Rule 5 | V42 disaster (0.943) |
| Gold-58 = sanity only (7 fractures, 9 MCL) | AGENTS.md Rule 6 | Theorem 7 |
| Training = isolated script + grad checkpointing + chunk≤4 | AGENTS.md Rule 7 | V34-V36 crashes |
| No credentials in code/logs/prompts | AGENTS.md Rule 8 | .gitignore + env vars |
| State only verified facts ("unverified" if not run) | AGENTS.md Rule 9 | Log evidence field |
| Team quota 5/day resets 00:00 UTC | AGENTS.md Rule 10 | `LIVE_STATE.md` |
| 384px weight ≥ 70%, no re-blending pre-blended | AGENTS.md Rule 11 | ADR-048, guardrail |

---

## 📝 LOGGING PROTOCOL (After EVERY Meaningful Action)

```bash
# Template - fill in brackets:
python tools/log_step.py \
  --agent "<YOUR_MODEL_NAME>" \
  --kind <STEP|DECISION|FAILURE|SUBMIT|SCORE|TRAIN> \
  --action "<what you did>" \
  --result "<observed OUTCOME, not intent>" \
  --evidence "<command output / file path / ref ID>" \
  --next "<SINGLE next actionable step>"

# THEN ALWAYS:
python tools/state_snapshot.py
```

### Log Kind Definitions:
| Kind | When to Use | Example Result |
|------|-------------|----------------|
| **SUBMIT** | Kaggle competition submission | "Ref 57443706 PENDING" |
| **SCORE** | Public LB score received | "Public LB: 0.961" |
| **TRAIN** | Training start/finish/checkpoint | "Engine 12 Epoch 8 complete, Gold AUC 0.8808" |
| **DECISION** | Strategy change, ADR created | "ADR-049 accepted; V49 OOF-calibrated path" |
| **FAILURE** | Error, crash, regression | "V47 0.953: ADR-048 double-blending root cause" |
| **STEP** | Any other meaningful action | "Built V49 notebook; guardrail PASSED" |

---

## 🛡️ PRE-FLIGHT CHECKLIST (Before EVERY Kaggle Push)

```bash
# 1. Guardrail validation (MANDATORY - exit code 0 required)
python tools/validate_kernel_guardrail.py <notebook.ipynb>

# 2. Submission integrity
python tools/verify_submission_integrity.py <submission.csv>

# 3. Quota check
python tools/state_snapshot.py
# Verify "Used today: X / 5 — Y remaining"

# 4. Selection verification (manual in Kaggle UI)
# Slot 1: 57253840 (0.958) CHECKED
# Slot 2: <experimental> CHECKED
```

**If ANY check fails → DO NOT SUBMIT. Fix first.**

---

## 📦 ASSET VERIFICATION (Run on Agent Switch)

```bash
# Verify all 10 foundational assets are live:

# Asset 1: Constitution
cat AGENTS.md | head -20

# Asset 2: Telemetry
python tools/state_snapshot.py && cat docs/LIVE_STATE.md

# Asset 3: Audit Trail
tail -5 docs/LIVE_STEP_LOG.md

# Asset 4: Guardrail
python tools/validate_kernel_guardrail.py submissions/v48_apex_quad_arm_ensemble/...

# Asset 5: LB Daemon
ps aux | grep monitor_v48_scoring.py

# Asset 6: API Ingest
python tools/state_snapshot.py

# Asset 7: Engine 12 Weights
kaggle datasets files cyruspinto/rsna-knee-engine12-convnext-large-weights

# Asset 8: Engine 13 Weights
kaggle datasets files cyruspinto/rsna-knee-engine13-swin-base-weights

# Asset 9: Fold 1 Notebooks
python -m py_compile ~/Downloads/colab_engine12_fold1_master.ipynb
python -m py_compile ~/Downloads/colab_engine13_fold1_master.ipynb

# Asset 10: V48 Ensemble
ls submissions/v48_apex_quad_arm_ensemble/
```

---

## 🚨 SESSION END PROTOCOL (Before Handoff)

```bash
# 1. Final state snapshot
python tools/state_snapshot.py

# 2. Log exact resume point
python tools/log_step.py \
  --agent "<YOUR_MODEL>" \
  --kind STEP \
  --action "Session handoff" \
  --result "All docs synchronized; V48 monitoring active; Colab Fold 1 notebooks ready" \
  --evidence "docs/LIVE_STATE.md; task-20791; ~/Downloads/colab_engine12_fold1_master.ipynb; ~/Downloads/colab_engine13_fold1_master.ipynb" \
  --next "Resume: Monitor V48 score via task-20791 → if ≥0.9610 select Slot 2 → else run parallel Fold 1 training on Colab Tabs 1&2 → build V49 with OOF weights"

# 3. Git commit (docs only - NEVER .env, weights, data)
git add docs/ decisions/ tools/ AGENTS.md
git commit -m "docs: session handoff $(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

---

## 🎯 CURRENT RESUMPTION STATE (As of 2026-10-20 17:30 UTC)

| Item | Status | Action Required |
|------|--------|-----------------|
| **V48 Submission** | `57443706` PENDING (queued ~5.5h) | Monitor `task-20791` for score |
| **Slot 1 (Safety)** | `57253840` 0.958 LOCKED | Verify in Kaggle UI — NEVER TOUCH |
| **Slot 2 (Experimental)** | Empty | Await V48 score ≥ 0.9610 |
| **Engine 12 Weights** | LIVE on Kaggle (0.8808 Gold) | Ready for multi-fold training |
| **Engine 13 Weights** | LIVE on Kaggle (0.8577 Gold) | Ready for multi-fold training |
| **Colab Tab 1 (T4)** | Engine 13 Fold 1 notebook ready | Run `colab_engine13_fold1_master.ipynb` |
| **Colab Tab 2 (A100)** | Engine 12 Fold 1 notebook ready | Run `colab_engine12_fold1_master.ipynb` |
| **Daily Quota** | 1 / 5 remaining today | Reserve for V49 if V48 < 0.9610 |
| **Next ADR Number** | ADR-050 | For next strategic decision |

---

## 💡 WHY THIS PROTOCOL WORKS

1. **No Trust Required:** Every number comes from `state_snapshot.py` → Kaggle API. No agent memory needed.
2. **No Hallucination Possible:** Hard rules in `AGENTS.md` are enforced by `validate_kernel_guardrail.py` (machine).
3. **No Context Loss:** `LIVE_STEP_LOG.md` is append-only, immutable, complete history.
4. **No Re-Work:** 10 Long-Run Assets survive disk wipes, VM recycles, agent amnesia.
5. **No Ambiguity:** Canonical doc chain has single read order. `LIVE_STATE.md` is ground truth.
6. **No Coordination Overhead:** Single-operator lock. Final day = mechanical checklist execution.

---

**This protocol has been battle-tested across 10+ agent switches.**  
**It works. Use it. Every time.**

---

*End of Autonomous Agent Continuity Protocol v1.0*  
*Part of the 10 Long-Run Foundational Assets (Asset 10)*  
*Enforced by AGENTS.md Session Start Checklist*