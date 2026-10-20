# Risk Register — Everything That Can Stop Us Reaching Top 10 / #1

**Created:** 2026-10-20 06:35 UTC · **Owner:** whichever agent is active (see `AGENTS.md`)
**Live numbers:** always in [LIVE_STATE.md](LIVE_STATE.md). Snapshot at creation: rank **688 / 6,255**, score **0.958**, top-10 cutoff **0.967**, #1 **0.969**, **1,542 teams ≥ 0.958**, ~65 h left.

> [!IMPORTANT]
> **Honest framing.** Nobody can guarantee Top 10 or #1. The gap is **+0.009 AUC to Top 10** and **+0.011 to #1**, with ~65 h, ~13 submissions left (≈4 today + 5 + 5 minus safety margin) and the public LB rounded to 3 decimals. This register exists to maximize the *probability* of a jump and to make sure we never lose ground through avoidable errors. Each item lists: **what goes wrong → how we detect it → how we prevent it**. Items marked ✅ have already hurt us in this project.

---

## A. Strategic risks (the biggest ones)

| # | Risk | Detection | Prevention |
|---|---|---|---|
| A1 ✅ | **0.958 is the public-notebook plateau** (1,542 teams tie at ≥0.958). Anything built by re-blending public weights lands here. | LB distribution in LIVE_STATE. | Only *new signal* moves us: newly trained models (E12/E13 + more folds), better labels, new inputs. Stop re-weighting existing arms. |
| A2 ✅ | **Pseudo-diversity**: V18–V33 spent 3 days re-weighting the same 2 models (0.956–0.958). | Cross-model rank correlation > 0.95 on OOF. | Compute OOF correlation before blending; reject arms with ρ > 0.95 to existing blend. |
| A3 | **Efficiency-prize constraint (<500 s) caps AUC.** Main LB allows up to 9 h; top teams likely run much larger ensembles. | Compare our runtime vs AUC frontier. | Decouple: final slot for *max AUC* may use up to ~8 h runtime. Efficiency prize is a separate bet, not a cap on the main bet. |
| A4 | **Single-fold new models** (E12/E13 trained on one split) are noisy; one fold ≈ high-variance estimator. | OOF AUC per fold spread > 0.01. | Prefer ≥3 folds per new architecture before giving >15 % weight. |
| A5 | **Too few submissions left for exploration** (~13). Each wasted submission = lost information. | Quota in LIVE_STATE. | Every submission must test exactly one hypothesis with a predicted effect written in the log *before* submitting. |
| A6 | **Public LB overfitting / shake-up.** Public LB is a subset; differences of 0.001 are noise. | Private LB after deadline only. | Choose final slots by OOF CV + public jointly, never public alone; prefer the more diverse, more robust candidate. |
| A7 | **Rounding hides progress**: public LB shows 3 decimals; a +0.0004 gain is invisible. | Same score as baseline. | Judge small changes by OOF CV, not LB. Use LB only for ≥0.002 effects. |
| A8 | **Chasing #1 at the cost of the medal floor.** Wild final-day changes can drop both slots. | — | Two-track: one safety slot, one ambition slot. Freeze code 12 h before deadline. |
| A9 | **Time-sink on infrastructure/docs instead of score.** | Hours since last new-signal experiment. | Docs are automated (snapshot + log tool). Spend ≥80 % of remaining time on training/inference. |
| A10 | **Teammate actions** (5-member team): someone submits, consumes quota, or changes final selection. | Unexpected entries in submissions list. | Announce plan in team chat; one person owns final selection; check LIVE_STATE before each submit. |

## B. Modeling risks

| # | Risk | Detection | Prevention |
|---|---|---|---|
| B1 ✅ | **Gold-58 overfitting** (7 fractures, 9 MCL). One flip ≈ 0.003 class AUC. | Gold-58 up, LB flat/down (V7–V11). | Gold-58 = sanity only. Tune on full 5-fold OOF (4,407 studies). |
| B2 ✅ | **Hand-crafted clinical priors** (ACL→contusion boosts). V12 fell to 0.953. | LB drop after adding priors. | **Do not ship the "kinematic coupling" boosts** in the V47 blueprint unless OOF proves ≥ +0.002. |
| B3 ✅ | **Raw-probability blending** across families. | Blend AUC < best single arm on OOF. | Percentile ranks only. |
| B4 ✅ | **Power sharpening p>1** (V42 0.943). | — | Linear rank blend only. |
| B5 ✅ | **Over-weighting weak arms** (30 % to a 0.73 arm). | — | Weights from OOF optimizer; cap new arms at 15 % otherwise. |
| B6 | **Label noise** — training labels partly report-derived; noisy positives cap AUC. | Train loss plateaus high on specific classes. | Soft labels / label smoothing; consider report-derived soft targets (public "read the report" approach) for training only. |
| B7 | **Class imbalance on rare findings** (fracture, MCL, Baker's) dominates macro-AUC variance. | Per-class OOF AUC table. | Per-class weights in the blend; focal/asymmetric loss; oversample positives. |
| B8 | **Prevalence shift** train → public → private (organizers warned). | — | Rank-based blending is invariant to monotone shifts; avoid calibration-dependent tricks. |
| B9 | **Wrong series selection** (sagittal vs coronal vs axial, fat-sat vs not). | Per-plane ablation OOF. | Verify plane detection on a random 50-study sample; log plane counts at inference. |
| B10 ✅ | **Slice order inversion / sorting** (V1–V5). | Visual check of slice montage. | Sort by ImagePositionPatient projected on slice normal (C++ fastread). |
| B11 | **Laterality (left/right knee) flips** break medial/lateral classes. | Medial vs lateral OA AUC asymmetry. | Canonicalize laterality; flip-TTA averaged in rank space. |
| B12 ✅ | **44 vs 96 slice domain shift** for positional heads. | — | Permutation-invariant attention-MIL. |
| B13 | **Resolution mismatch** train 384 vs infer 224 (or vice versa). | Single-model OOF vs inference parity test. | Inference must reproduce training preprocessing exactly; test with 20 OOF studies end-to-end. |
| B14 | **Normalization mismatch** (ImageNet mean/std vs per-volume z-score). | Parity test fails. | Store preprocessing config next to weights (json) and load it in inference. |
| B15 | **BatchNorm/LayerNorm eval-mode mistakes** (model.train() at inference). | Non-deterministic outputs on repeat. | Assert-free check: run same study twice, log max diff. |
| B16 | **Fold leakage** (same patient in train & val) inflates OOF. | OOF ≫ LB. | Group folds by patient/study; reuse `generate_stratified_multilabel_folds.py`. |
| B17 | **Picking best epoch on the val fold** then reporting it as OOF → optimistic CV. | — | Note it; compare arms only under the same protocol. |
| B18 | **External data disqualification** (OAI/fastMRI debates). | Forum rulings. | Only official data unless explicitly allowed (ADR-017). |
| B19 | **Pseudo-labelling the test set** — prohibited/risky in code comps. | — | Do not. |
| B20 | **Too-short training** (8 epochs single fold) under-fits large backbones. | Val AUC still rising at last epoch. | Log per-epoch AUC; extend if still rising and compute allows. |
| B21 | **Model collapse** — all 12 heads output near-identical ranks. | Mean off-diagonal head correlation > 0.85. | Already measured for Engine 5 (0.449). Check for every new engine. |
| B22 | **TTA that hurts** (flip on asymmetric anatomy without label swap). | OOF with/without TTA. | Only keep TTA proven on OOF. |

## C. Training-infrastructure risks (Colab)

| # | Risk | Detection | Prevention |
|---|---|---|---|
| C1 ✅ | **CUDA OOM** on T4 with Swin-384 (14.4 GB). | Traceback. | grad checkpointing + chunk_size 4 + k=8. |
| C2 ✅ | **Zombie tensors** after in-notebook crash. | VRAM not freed. | `!python train_x.py` isolated process. |
| C3 ✅ | **f-string / syntax errors** in generated scripts (line 246). | SyntaxError at launch. | `python -m py_compile train_x.py` locally before giving cells to the user. |
| C4 ✅ | **Drive mount hang**. | Cell stalls on auth. | Save to /content, push via Kaggle CLI, Drive optional. |
| C5 | **Colab disconnect / runtime recycle** (idle or 12–24 h limit) loses unsaved weights. | Session gone. | Save + push checkpoint **every epoch**, not only at end. |
| C6 | **Compute-unit exhaustion** (A100 burns fast). | Colab usage page. | Budget per run in the log; T4/L4 for long runs, A100 only for large models. |
| C7 | **HF Hub rate limit / download failure** for timm weights. | "unauthenticated requests" warning, timeouts. | Set HF_TOKEN via Colab Secrets; cache weights to a Kaggle dataset. |
| C8 | **Kaggle dataset version push fails silently** (`create || version`). | `kaggle datasets status`. | After push, run `kaggle datasets files <slug>` and log file size. |
| C9 | **Library version drift** (timm/torch) between Colab and Kaggle inference. | Missing keys on load_state_dict. | Pin versions; load with strict=True in a local parity test; record versions with weights. |
| C10 | **Python 3.13 on Colab vs older on Kaggle** → pickled objects incompatible. | Unpickling error. | Save `state_dict` only (no full-model pickle). |
| C11 | **Mixed precision NaN** (fp16 on T4). | Loss = nan. | GradScaler; bf16 on A100; nan_to_num on logits. |
| C12 | **Wrong checkpoint uploaded** (last vs best). | File timestamp/metrics json mismatch. | Write `metrics.json` (epoch, val AUC) beside each checkpoint. |
| C13 | **RAM (not VRAM) OOM** from caching full volumes. | Kernel restarts. | Stream from disk; limit DataLoader workers to 2 on Colab. |

## D. Submission / Kaggle-runtime risks

| # | Risk | Detection | Prevention |
|---|---|---|---|
| D1 ✅ | **Fatal asserts on hidden test** (V34–V36). | "Notebook threw exception". | Guardrail AST scan; fail-soft median fallback. |
| D2 | **Hidden-test runtime ≠ interactive runtime.** 496 s was on 3 studies; hidden test is ~1,300+ studies. | Kaggle submission runtime field. | Estimate per-study time × N; never claim efficiency eligibility from 3-study runs. |
| D3 | **9 h timeout** for heavy ensembles. | Submission error "timeout". | Per-study time budget; checkpoint partial predictions; skip arms if over budget (fail-soft). |
| D4 | **Internet disabled** in submission → pip install / HF download fails. | Error at import. | All wheels and weights as Kaggle datasets attached to the notebook. |
| D5 | **Dataset not attached / wrong version attached.** | FileNotFound → fallback predictions → low score. | Print and log the resolved weight paths + sha256 at start of kernel. |
| D6 | **Silent fallback** (arm fails, median prior used) → valid submission with bad score. | Score drop without error. | Count fallbacks; write them to the kernel log; if fallbacks > 0 on interactive run, do not submit. |
| D7 | **Submission format errors** (column order, UID dtype, missing rows). | Scoring error. | `tools/verify_submission_integrity.py` against sample_submission. |
| D8 | **GPU memory on 2×T4** different from Colab GPU. | OOM in Kaggle only. | Batch size by device memory; test on Kaggle interactive first. |
| D9 | **Kernel version confusion** (submitting an older version of the notebook). | Description/version mismatch. | Put candidate ID in notebook title and in a printed banner; log kernel version number. |
| D10 | **Quota exhaustion near deadline** (5/day, team-wide, resets 00:00 UTC = 05:30 IST). | LIVE_STATE. | Keep 1 submission spare each day; final-day plan uses ≤3. |
| D11 | **Scoring queue delays** on final day (hours). | PENDING status. | Last ambition submission no later than **2026-10-22 12:00 UTC**. |
| D12 | **Forgetting to select final 2** → Kaggle auto-selects best-public, which may not be the intended pair. | Kaggle UI. | Manual selection check at T-24 h and T-6 h, logged with screenshot. |
| D13 | **Efficiency-prize ineligibility**: only selected submissions count; must beat baseline. | Rules. | If going for efficiency prize, the efficient candidate must be one of the 2 selected. |
| D14 | **Wrong efficiency formula used in planning.** Official (ADR-032): `E = (AUC−M)/(B−M) − T/32400`. | — | Fixed in the guardrail skill on 2026-10-20. |

## E. Process / agent risks (switching AI models)

| # | Risk | Detection | Prevention |
|---|---|---|---|
| E1 ✅ | **Stale facts propagated** (docs said "rank 583", "slot 1 = V6"; reality differs). | Snapshot disagrees with doc. | Numbers only from LIVE_STATE.md (auto-generated). |
| E2 ✅ | **Hallucinated or inflated claims** in docs ("World #1", unverified formulas, unverified runtimes). | Claim lacks evidence line. | Every number in a log entry needs `--evidence`. |
| E3 | **Contradicting ADRs** (e.g. V47 blueprint adds priors that ADR-002 banned). | Reviewer reads ADR index. | New ADR must cite the ADRs it supersedes. |
| E4 | **Context loss on agent switch / compaction.** | New agent re-asks basics. | AGENTS.md read order + LIVE_STEP_LOG tail. |
| E5 | **Agent edits wrong file / overwrites working notebook.** | git diff. | Commit before each risky change; new candidates in new folders. |
| E6 | **Credential leak** (.env, kaggle.json in git or prompts). | `git log -p | findstr key`. | `.gitignore` covers `.env`; ADR-005. |
| E7 | **Background monitors referencing old candidates** (task watched V42/V43 long after). | Heartbeat prompt mentions stale refs. | Recreate monitors per candidate; kill stale ones. |
| E8 | **Doc sprawl**: 4 competing handoff files (HANDOFF.md, MASTER_HANDOFF.md, RSNA_Knee_Handoff_v2.md, ULTIMATE…). | — | Canonical chain = AGENTS.md → LIVE_STATE → LIVE_STEP_LOG → this register → Encyclopedia. Older handoffs are historical. |
| E9 | **Unverified code handed to user** (syntax errors in Colab). | User-reported crash. | `py_compile` + 1-batch dry run on CPU before handing over. |
| E10 | **Fatigue decisions at night (IST)** close to deadline. | — | Pre-written final-day checklist (Encyclopedia §9); follow it mechanically. |

## F. Rules / compliance risks

| # | Risk | Prevention |
|---|---|---|
| F1 | Private sharing of code outside the team. | Only within team or public notebooks. |
| F2 | Team-merge or multi-account violations (another account named "Cyrus" exists on LB — not ours). | One account per person; never submit from other accounts. |
| F3 | Winner obligations (open-source licence, write-up, reproducibility). | Keep training scripts, seeds, versions committed. |
| F4 | Unapproved external data/models. | ADR-017; check forum rulings before using any new pretrained medical weights. |

---

## Update protocol
- New risk discovered → add a row (next ID in its section) + log a `FAILURE` or `DECISION` step.
- Risk materialized → mark ✅ and add the incident to the Encyclopedia timeline.
