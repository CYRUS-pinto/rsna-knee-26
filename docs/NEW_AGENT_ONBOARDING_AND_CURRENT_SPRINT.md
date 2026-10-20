# New Agent Onboarding & Current Sprint Briefing

**Document:** Canonical Master Handoff & Sprint State  
**For:** Any incoming AI Model (Gemini, Claude Code, OpenAI Codex/GPT-5/o3, Cursor, Antigravity) or Human Teammate  
**Updated:** 2026-10-20 16:10 UTC  
**Deadline:** 2026-10-22 23:59 UTC (**~55.8 hours remaining**)  
**Live Numbers:** Always regenerate and trust [docs/LIVE_STATE.md](LIVE_STATE.md) via `python tools/state_snapshot.py`.

---

## 1. Quick Orientation: What You Need to Know in 60 Seconds

1. **Competition:** [RSNA Knee Abnormality Detection (Kaggle)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection). 12 binary abnormality targets.
2. **Metric:** Macro-Averaged ROC AUC across the 12 targets + $18,000 Efficiency Prize (<500s dual-GPU runtime).
3. **Current Position:** Rank **737 / 6,274** with Public PB **0.958**.
   - Exactly 1,601 teams tie at $\ge 0.958$ (this is the **public notebook plateau**).
   - **Top 10 cutoff:** $\ge 0.967$ (+0.009 gap).
   - **#1 cutoff:** $0.969$ (+0.011 gap).
4. **Quota:** 5 submissions per day (team-wide, 5 members). Resets at 00:00 UTC (05:30 IST).
   - **4 / 5 used today. 1 submission remaining.**
5. **The Sacred Two-Track Defense:**
   - **Slot 1:** Permanently locked to Ref `57253840` (Candidate V17 / V45 / 0.958 baseline). **NEVER DESELECT OR OVERWRITE THIS SLOT** unless an experimental candidate achieves $\ge 0.9610$ on the public LB.
   - **Slot 2:** Dedicated to experimental candidates (currently evaluating V48).

---

## 2. What is COMPLETE & VERIFIED (Do NOT Repeat)

| Component / Asset | Status | Verification & Evidence | Notes / Key Takeaways |
|---|---|---|---|
| **Engine 12 (ConvNeXt-Large-384)** | **Trained & Uploaded** | `cyruspinto/rsna-knee-engine12-convnext-large-weights` (`engine12_convnext_large_best.pt`, 788.3 MB) | Peak Gold-58 Macro AUC: **0.8808**. Standouts: ACL > 0.92, MedOA > 0.96. Gated Attention-MIL. |
| **Engine 13 (Swin-Base-384)** | **Trained & Uploaded** | `cyruspinto/rsna-knee-engine13-swin-base-weights` (`engine13_swin_base_best.pt`, 348.8 MB) | Peak Gold-58 Macro AUC: **0.8577**. Standouts: Fracture 0.9042, MedOA 0.9581. Asymmetric Focal Loss. |
| **Candidate V46** | **Scored** | Ref `57426806` scored **0.957** | Multi-planar spatial decoupling tested; showed clean stability but no breakthrough over 0.958. |
| **Candidate V47** | **Scored & Diagnosed** | Ref `57429824` scored **0.953** | **CRITICAL REGRESSION:** Diagnosed in ADR-048. Double-blending `raptor224_sub.csv` ballooned low-res 224px weight to 57.8%, causing spatial Nyquist blur on micro-tears. |
| **Candidate V48** | **Submitted & Evaluating** | Ref `57443706` submitted 15:22 UTC (**PENDING**) | **Apex Quad-Arm Sovereign Ensemble.** Fixed ADR-048: enforces $\ge 70\%$ native 384px dominance, zero redundant 224px blending. Passed pre-flight AST guardrail. |
| **Notebook Bug Patch** | **Cleaned & Committed** | Commit `78e820e` | Fixed a silent 13-column typo on Gold-58 line 117 across all 4 Colab notebooks. |
| **Autonomous Monitor** | **RUNNING** | Background task `task-154` (`tools/autonomous_overnight_monitor.py`) | Actively polling Kaggle API every 60s for Ref 57443706. Auto-logs and updates state when scored. |

---

## 3. What is IN-PROGRESS Right Now

1. **Candidate V48 Evaluation on Kaggle:**
   - Tracking Kaggle Submission Ref `57443706`.
   - Polling worker status via background daemon `task-154`.
2. **Multi-Fold Training in Colab:**
   - Both Engine 12 and Engine 13 were trained on **Fold 0 only**.
   - Training remaining folds (Folds 1–4) eliminates single-fold sampling variance ($\pm 0.004$ AUC) and enables multi-fold OOF ensembling.
   - Notebooks are ready for one-click execution:
     - [colab_swin_single_click_master.ipynb](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/colab_swin_single_click_master.ipynb) (T4 / L4 GPU)
     - [colab_engine12_convnext_large_master.ipynb](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/colab_engine12_convnext_large_master.ipynb) (A100 GPU)

---

## 4. Exactly What to Do NEXT (Action Plan for New Agent)

Follow this precise sequence upon taking over:

```
Step 1: Check V48 Score
   │
   ├── If V48 >= 0.959:
   │      Lock V48 in Slot 2 (Kaggle Submissions UI).
   │      Never touch Slot 1 (Ref 57253840).
   │      Prepare Candidate V49 with 5-fold OOF calibrated weights.
   │
   └── If V48 < 0.959:
          Preserve Slot 1.
          Inspect delta vs V47 (0.953) and V45 (0.958).
          Proceed with Multi-Fold Engine 12 + Swin-Base + DINOv2 integration.
```

### Action 1: Check the Status of V48
Run this command in the shell:
```bash
python -c "import kaggle; api = kaggle.KaggleApi(); api.authenticate(); [print(s.ref, s.date, s.status, getattr(s, 'public_score', None)) for s in api.competition_submissions('rsna-knee-abnormality-detection')[:2]]"
```

### Action 2: Verify Colab Training Runs
Check with the user if Tab 1 (Swin-Base-384) or Tab 2 (ConvNeXt-Large-384) has completed epochs. If completed, verify that the new fold weights were uploaded to Kaggle Datasets via:
```bash
kaggle datasets files cyruspinto/rsna-knee-engine12-convnext-large-weights
kaggle datasets files cyruspinto/rsna-knee-engine13-swin-base-weights
```

### Action 3: Pre-Flight Guardrail Check Before ANY Push
Before pushing any kernel or submission to Kaggle, you MUST run:
```bash
python tools/validate_kernel_guardrail.py <path_to_notebook.ipynb>
```
If it does not return **`PASSED`**, DO NOT PUSH. Fix the violation first.

### Action 4: Log Every Single Meaningful Step
After every action, run:
```bash
python tools/log_step.py --agent "<Your Model Name>" --kind <STEP|DECISION|FAILURE|SUBMIT|SCORE|TRAIN> \
  --action "<what you did>" --result "<observed result>" \
  --evidence "<command / file / ref id>" --next "<next step>"
```

---

## 5. The 11 Immutable Hard Rules (Violating Any of These Will Cost Score)

| # | Rule | Why It Exists / What Went Wrong Historically |
|---|---|---|
| **1** | **Never deselect the 0.958 safety submission (Ref `57253840`).** | Prevents zero-score disaster. Only replace if a candidate achieves $\ge 0.9610$ on public LB. |
| **2** | **Rank-percentile blending only.** | Arithmetic probability averaging and power sharpening ($p > 1$) collapsed V42 to 0.943. ROC AUC depends strictly on rank monotonicity. |
| **3** | **No hand-tuned clinical "boost" priors.** | Manual conditional boosts (e.g. ACL boosting Bone Contusion) dropped V12 to 0.953. Only use priors proven on 4,407 full OOF. |
| **4** | **No fatal asserts / raises on test data.** | Hidden test set has corrupted tags or missing series. Hard asserts crashed V34–V36. Always use try/except with column-median fallback. |
| **5** | **New model weight $\le 15\%$ until OOF weights exist.** | Protects against uncalibrated high-variance models swamping the anchor. |
| **6** | **Gold-58 is a sanity check, never a tuning target.** | Gold-58 has only 7 fractures and 9 MCL tears ($\pm 0.035$ swing per sample). Tune only on full OOF. |
| **7** | **Training = isolated script (`!python train_x.py`).** | In-notebook training leaves zombie VRAM tensors and crashes Colab. Always set `chunk_size <= 4` and `set_grad_checkpointing(True)`. |
| **8** | **No credentials in files, notebooks, or git.** | Use environment variables or Colab Secrets (`userdata.get`). |
| **9** | **State only what you verified.** | Never say "will score 0.965" or "efficiency is 400s" without empirical evidence. If unrun, state "unverified". |
| **10** | **Quota is team-wide (5 members, 5 subs/day).** | Resets at 00:00 UTC (05:30 IST). Always check [docs/LIVE_STATE.md](LIVE_STATE.md) before submitting. |
| **11** | **Resolution Dominance ($\ge 70\%$ native 384px) & Zero Redundant Blending.** | V47 fell to 0.953 because double-blending low-res 224px slices caused Nyquist blur on meniscal/fracture tears (ADR-048). Enforced automatically by `tools/validate_kernel_guardrail.py`. |

---

## 6. Canonical Documentation Index

1. [AGENTS.md](../AGENTS.md) — Mandatory protocol read first by all agents.
2. [docs/LIVE_STATE.md](LIVE_STATE.md) — Dynamic facts: rank, quota, submissions, hours left.
3. [docs/LIVE_STEP_LOG.md](LIVE_STEP_LOG.md) — Full trajectory of every action, observed result, and `--next`.
4. [docs/FINAL_DAY_CHECKLIST.md](FINAL_DAY_CHECKLIST.md) — Hour-by-hour operational timeline (T-56h to T-0h).
5. [docs/MODEL_REGISTRY.md](MODEL_REGISTRY.md) — Index of all trained weights, resolutions, and dataset slugs.
6. [docs/RISK_REGISTER_TOP10.md](RISK_REGISTER_TOP10.md) — Complete failure modes (Sections A through G).
7. [decisions/ADR-049-ceiling-engineering-forensics-and-top1-limitations.md](../decisions/ADR-049-ceiling-engineering-forensics-and-top1-limitations.md) — Architectural decision record on ceiling engineering.
8. [docs/CEILING_ENGINEERING_AND_FAILURE_PREVENTION_MANUAL.md](CEILING_ENGINEERING_AND_FAILURE_PREVENTION_MANUAL.md) — Grandmaster failure prevention manual and 10 permanent assets.
