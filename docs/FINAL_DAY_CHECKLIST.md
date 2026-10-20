# Final 56-Hour Execution & Selection Checklist (T-56h to T-0h)

**Competition:** RSNA 2026 Knee Abnormality Detection  
**Official Deadline:** 2026-10-22 23:59 UTC (2026-10-23 05:29 IST)  
**Strict Rule:** Every agent, regardless of architecture, MUST follow this exact sequence. No exceptions.

---

## 1. Submission Quota Budget & Cadence

| Window | Remaining Quota | Planned Submission Action | Goal / Hypothesis |
|---|---|---|---|
| **Day 1 (Oct 20) T-56h to T-48h** | 1 sub left (4/5 used) | Ref 57443706 (Candidate V48) currently evaluating. Keep final slot in reserve unless V48 errors. | Quad-Arm Sovereign (Engine 12 + Swin-B + E11 + Super-CoAtNet) testing 384px dominance. |
| **Day 2 (Oct 21) T-48h to T-24h** | 5 subs available | **Sub 1 (T-42h):** V49 OOF-Optimized Calibrated Quad-Arm.<br>**Sub 2 (T-36h):** V50 Multi-Fold Engine 12 (ConvNeXt-Large folds 0+1+2).<br>**Sub 3 (T-30h):** V51 Multi-Plane Cross-Attention Fusion.<br>**Sub 4 (T-24h):** V52 Diversity Titan (DINOv2 / ConvNeXt / Swin).<br>**Sub 5 (T-24h):** RESERVE (Do not burn). | Move beyond the 0.958 public notebook plateau into 0.960+ territory. |
| **Final Day (Oct 22) T-24h to T-0h** | 5 subs available | **Sub 1 (T-18h):** V53 Apex 5-Fold Sovereign Master.<br>**Sub 2 (T-12h):** V54 Final Ambition Submission (Best CV).<br>**Sub 3 (T-8h):** LAST CALL SUBMISSION (Queue delays can be 4+ hours).<br>**Sub 4 & 5:** EMERGENCY ONLY. | Reach top percentile while locking down safe medal floor. |

---

## 2. Hard Pre-Flight Guardrail Checklist (Before EVERY Kernel Push)

Before running `kaggle kernels push` or submitting to Kaggle, you MUST verify:

- [ ] **1. Run AST Guardrail:** `python tools/validate_kernel_guardrail.py <notebook_path>` returns **PASSED**.
- [ ] **2. Resolution Dominance Rule:** $\ge 70\%$ of model weights belong to native $\ge 384\text{px}$ backbones. Low-res ($224\text{px}$) components MUST NOT exceed $30\%$.
- [ ] **3. Zero Redundant Blending:** Multi-scale files (e.g. `raptor_sub.csv`) are NEVER blended with constituent single-scale files (e.g. `raptor224_sub.csv`).
- [ ] **4. Rank Percentile Only:** No raw probability averaging, no power sharpening ($p > 1$), no exponential transforms.
- [ ] **5. Fail-Soft Exception Handling:** Zero unconditional `assert` statements on test dataframe lengths or feature presence. Fallback median prior imputed on any slice read error.
- [ ] **6. Offline Asset Verification:** All model weights (`.pt`, `.pth`, `.bin`) and wheels are confirmed present in Kaggle input datasets. No network calls during inference.
- [ ] **7. Runtime Estimation:** Total inference time across test set verified $\le 5$ hours (well under the 9-hour hard timeout).

---

## 3. Final Submission Selection Protocol (Critical)

> [!CAUTION]
> Kaggle allows **EXACTLY TWO (2)** selected submissions for private leaderboard evaluation. If you do not manually select them, Kaggle will auto-select the highest public LB scores, which risks public LB overfitting and catastrophic shakeup.

### Slot 1: The Sovereign Safety Floor
- **Selected Submission:** Ref `57253840` (Candidate V17 / V45 / 0.958 baseline)
- **Constraint:** **NEVER DESELECT THIS SUBMISSION** unless an explicit new candidate achieves **$\ge 0.9610$** on the public LB.
- **Rationale:** Guarantees medal/ranking baseline. Prevents zero-score disaster.

### Slot 2: The Ambition / Best-OOF Candidate
- **Selected Submission:** The candidate with the highest validated **5-fold OOF Macro-AUC** (not solely public LB score).
- **Selection Deadline:** Must be locked in Kaggle UI by **2026-10-22 18:00 UTC** (6 hours before cutoff).
- **Verification:** Both team members must log in to the Kaggle Submissions page, visually inspect the two radio buttons/checkboxes, and verify that the selected IDs match `docs/LIVE_STATE.md`.

---

## 4. T-Minus Operational Timeline

### T-48 Hours (2026-10-20 23:59 UTC / Reset 1)
- [ ] Check V48 score and log to `LIVE_STEP_LOG.md`.
- [ ] Snapshot competition leaderboard: `python tools/state_snapshot.py`.
- [ ] Daily quota reset to 5/5.
- [ ] Verify Colab training jobs for remaining folds of Engine 12 (ConvNeXt-Large) and Engine 13 (Swin-Base).

### T-24 Hours (2026-10-21 23:59 UTC / Reset 2)
- [ ] All model training stops. Code freeze on training pipelines.
- [ ] Complete 5-fold OOF weight optimization (Nelder-Mead / scipy.optimize.minimize on OOF ranks).
- [ ] Snapshot competition leaderboard: `python tools/state_snapshot.py`.
- [ ] Daily quota reset to 5/5.
- [ ] Confirm at least 2 clean submissions ready for final day testing.

### T-12 Hours (2026-10-22 12:00 UTC)
- [ ] Push final ambition candidate (V54) to Kaggle kernels.
- [ ] Verify kernel is actively RUNNING on Kaggle backend.
- [ ] Do NOT launch new training runs; focus exclusively on inference stability and UI verification.

### T-6 Hours (2026-10-22 18:00 UTC) — Submission Lockdown
- [ ] Ensure all submissions are scored and complete.
- [ ] Navigate to Kaggle Submissions page.
- [ ] Select **Slot 1 (Safety 0.958)** and **Slot 2 (Best Ambition / CV)**.
- [ ] Take a browser screenshot of the selected submissions.
- [ ] Save screenshot proof in repository artifacts / docs.
- [ ] Run `python tools/state_snapshot.py` to archive final numbers.

### T-1 Hour (2026-10-22 23:00 UTC) — Final Verification
- [ ] Confirm no teammate made accidental overrides.
- [ ] Confirm no pending submissions can overwrite selections.
- [ ] Generate final session handoff report.
