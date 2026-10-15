# RSNA Knee Abnormality Detection — Handoff v2 (with Issues Register)

**As of:** Thu Oct 15, 2026. Supersedes `RSNA_Knee_Handoff.md` (Oct 12). Paste this whole file into a new chat to continue.
**Labels:** [VERIFIED] = seen in pasted text/files or fetched directly. [UNVERIFIED] = claimed by someone, not seen. [ESTIMATE] = my inference.
**User:** beginner, Bengaluru (IST), works with an agentic IDE (Antigravity) that has CLI access to their Kaggle account and GitHub. Wants brutal honesty and a plan, not reassurance.

---

## 0. First actions for the next assistant (in this order)

1. **Make the GitHub repo private NOW.** It is PUBLIC [VERIFIED: fetched `github.com/CYRUS-pinto/rsna-knee-26` logged-out; page says "Public", 28 commits, 0 forks/stars]. Its history very likely contains `train.csv` (radiology report text + gold labels), `train_series.csv`, label files, and possibly a Kaggle token. See Issues S1–S2.
2. **Confirm the team merge** — deadline tonight, **Oct 15 11:59 PM UTC = 5:29 AM IST Oct 16**. No evidence in anything pasted that it happened (S3).
3. Ask the user for: V12 contents/score/runtime, real hidden-test runtime of V6, whether token is expired, Colab units balance, finals chosen (S15).

---

## 1. Competition facts [VERIFIED from pasted pages]

- Host RSNA, Kaggle code competition. 5,815 teams. Prizes $77k: 10 leaderboard prizes ($9k→$5k) + Efficiency track ($7k/$6k/$5k).
- 12 binary findings per study: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture. Metric = macro ROC-AUC (ranking only → rank-blending is natural).
- Data: 4,407 train studies, only **58 have gold labels**; the rest have multilingual reports only. ~1,300 hidden test studies, **no reports at test time**. 569.76 GB total (train + the 3-study example test); mounts only on Kaggle.
- Notebooks: ≤ 9 h GPU/CPU, internet off, output `submission.csv`. Public LB = ~30% of test; private = 70% (an agent doc says 60% — wrong).
- Dates (11:59 PM UTC): entry + team merger **Oct 15**; final submission **Oct 22**; winners' materials Nov 5.
- Rules: team ≤ 5; ≤ 5 submissions/day; 2 final selections; no private code/data sharing outside team; **Rule 4.b: do not publish/redistribute competition data; notify Kaggle if possible unauthorized access**; **Rule 3.6.b: competition code shared publicly must be shared on Kaggle (forum/notebooks)**; external data must be public, accessible, low-cost; winners: CC-BY-NC 4.0 code + public weights + video; identity verification required to submit.
- Host rulings seen: commercial/hosted LLMs on reports are allowed; KneeCoT prohibited; click-through datasets (e.g. OAI) not excluded per se but accessibility/licensing is the team's burden. **No ruling on OAI or OAI-trained weights.**

## 2. Leaderboard picture

- Oct 12 snapshot (CSV): 5,814 teams; ≥0.956: 122; ≥0.955: 186; ≥0.954: 837 (651 exactly 0.954). Top then 0.966. Per CLI later: top 0.967 (Tucker Arrants & Tony, CentralyDS), 0.966 Sergey Bryansky [UNVERIFIED, agent-reported].
- User's best 0.956 ≈ rank ~120 of 5,814 ≈ **top ~2%** [ESTIMATE; board has moved]. (MASTER_HANDOFF says "Top 12%, of 1,000+ teams" — wrong.)
- Medal cutoffs (gold ≈ top 20, silver ≈ top 265, bronze ≈ top 531) came from a third-party repo [UNVERIFIED].
- A 0.001 public difference moves hundreds of ranks near 0.954–0.956 but is within sampling noise on ~400 studies. Expect a private-board reshuffle.

## 3. User's submissions [VERIFIED from pasted Submissions list]

| Submission (notebook version) | Public | Note |
|---|---|---|
| rsna-knee-blend-gold (heliosli fork) v2 / fork v1 | 0.954 | baseline forks |
| CoAtNet S6 25D MIL 0957 v2 | 0.954 | |
| CoAtNet S6 25D MIL 0957 v6 | exception | hard assert `min rho < 0.99` tripped on hidden set |
| Quint Ensemble R7 Colab SWA v5 | 0.932 | agent's "Part B" reader (filename sort, missing JPEG-lossless decoder); effusion r≈−0.74 [agent-reported] |
| **v6** | **0.956** | clean dual engine, uniform 50/50 rank blend — **personal best** |
| v7 | 0.955 | asymmetric per-finding weights (gold 0.917→0.933, public −0.001) |
| v9 | 0.955 | + Colab "SWA 0.9167" arm (tri-engine) |
| v10, v11 | 0.955 | + "clinical priors" (+asymmetric in v11) |
| v12 (Kaggle) | running | description says "Super-Consensus V2 Weights … + Priors" |

- **V8 does not appear on Kaggle** (repo README lists it as 0.955 — unsupported).
- Finals: **0/2 selected**. Kaggle auto-selects if user doesn't.
- Key empirical lesson: **gold-58 gain of +0.016 produced −0.001 public.** Everything after v6 is noise-or-worse.

## 4. What the pipeline is

- **Stack (v6)**: Engine 1 = Raptor CoAtNet-384 SWA (`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`, nartaa; 96 slices in 5 fixed series slots, 140 mm crop, 384 px, 94 windows, anatomical-mirror averaging) **trained with OAI external knee data (per author's write-up + filename)** [VERIFIED]. Engine 2 = Heliosli ConvNeXt-S6 R7 bundle (`rsna-knee-compact-r7cnxfsss`); whether it used external data is [UNVERIFIED] (an agent asserts OAI/SKM-TEA — check the author's page). Fusion = per-finding percentile rank blend, 0.5/0.5 in v6.
- Public record of competition-only single-model ceiling: ~0.939–0.942; OAI-augmented 0.945–0.949 (nartaa write-up). So no clean (competition-only) final exists that matches 0.956.
- **Engine 3 / Colab training**: CoAtNet-2 384, trained on third-party precomputed corpus `dreaddevelopment/knee-raptor-corpus` (44 slices × 336 px, upsampled to 384) with agent-built "super-consensus" soft labels (from Qwen reports; "0.9172" gold AUC — tuned on the same 58 gold studies). First run (transformer head + ASL + head LR 1e-3 + OneCycle + GradScaler w/ bf16) **collapsed at epoch 2**: loss 0.42→1.65, gold AUC 0.6748→**exactly 0.5000** (tied/saturated scores). Agent patched: raw-logit AUC, divergence circuit-breaker, last-3-epoch SWA, no GradScaler for bf16, new recommended run `--head mil --loss bce --bs 8 --accum 1 --bb_lr 3e-5 --head_lr 3e-4 --workers 8 --epochs 8`. **Recommendation given: stop Engine 3** (public recipe of that corpus class scored ~0.924; V9 with the Colab arm didn't beat V6). Colab session was still billing (balance ~86→~83 units; ~6.77 units/h per agent) — disconnect.

## 5. Validation tooling (built in this chat, also in repo `tools/`/`validation/`)

- `gate_make_root.py`: builds a mock test root from the 58 gold studies (`test.csv`, `test_series.csv`, symlinked `test_series`, `gold.csv`). Needs a one-line `GATE_ROOT` override in the notebook's `find_test_root()` (+ any other root lookup).
- `gate_score.py`: per-class gold AUC; fails on any class < 0.60, inter-arm Spearman < 0.30, any `FALLBACK` in log, missing/non-finite rows; paired bootstrap vs uniform with 95% CI (Bonferroni 98.3% for 3 clusters). Clusters are a hypothesis: A = tears, B = OA, C = fluid/bone. Tested on synthetic data only.
- Purpose: **bug detector** (catches inverted effusion etc.), not an accuracy tuner. Expect 20–40 min, not 5.

---

## 6. ISSUES REGISTER

Severity: **C**ritical / **H**igh / **M**edium / **L**ow. Status: OPEN unless stated.

| ID | Sev | Issue | Evidence | Action |
|---|---|---|---|---|
| S1 | **C** | GitHub repo is PUBLIC; history likely contains `train.csv` (reports + gold labels), `train_series.csv`, labels. Rule 4.b exposure. Also Rule 3.6.b (public competition code belongs on Kaggle). | Fetched page: public, 28 commits. Agent log: `git rm --cached train.csv …` (so they were tracked & pushed). `git rm`/delete does NOT remove history. | Make private immediately. Then create a NEW private repo from a clean copy with no history (`git init`, no `--force` needed on an empty repo), delete the old repo. Check Insights → Traffic/Forks for clones. Decide with user whether to notify Kaggle (Rule 4.b). Cached copies elsewhere cannot be recalled. |
| S2 | **C** | Kaggle token (`KGAT_f10b…`) was hard-coded in Colab notebooks (cell 4; also as default in a regenerated notebook), pushed to GitHub, and pasted in chat. | Uploaded notebook; agent `sync_notebook.py` output. | **Expire it at kaggle.com/settings** (user action). Run `git log -S"KGAT_" --all` and grep `scratch/`, notebooks, Colab history/Drive. Use Colab Secrets going forward. Agent says it scrubbed the working tree; history is not scrubbed. |
| S3 | **C** | Team merge deadline tonight; status unknown; existence/verification of 4 teammates unknown. | No merge evidence pasted. | User must send/accept invites in Kaggle Team tab. Agents can't. Merge cap: submissions ≤ 5/day × days. Do not share code with non-members before merge. |
| S4 | **H** | Public README / MASTER_HANDOFF / ADRs contain false or unsupported claims (see §7). If repo stays public or a winner review happens, these damage credibility. | Fetched README; MASTER_HANDOFF.md. | Fix or delete the claims listed in §7 before any further sharing. |
| S5 | **H** | "Track 2: 100% competition-clean final" is advertised (README + badge "Zero External Data Track — Compliance 100%") but **does not exist**: Engine 3 collapsed, was stopped, and is a ~0.92-class model. Track 1 openly uses OAI-pretrained weights, contradicting the badge. | README; §4 above. | Remove the badge/claim or build the model. Decide the OAI policy explicitly (S6). |
| S6 | **H** | OAI risk unresolved: Raptor (in every candidate) is OAI-trained; host has not ruled; forum threads open. All three plausible finals share this risk. | nartaa write-up; forum threads. | Team decision: accept risk knowingly vs. wait for a ruling. Watch the thread through Oct 22. Prize-level submissions get audited; mid-ranks rarely do. |
| S7 | **H** | Real hidden-test runtime unknown. "438/450/413 s" is the **commit run on the 3-study sample**, repeatedly presented as hidden-test runtime and "Efficiency-prize compliant". Early baseline submission was pending > 3 h. Efficiency formula text is **missing from the pasted page** (equation image stripped); agent-invented formula "0.1·max(0,S−0.9)/0.1" is unsupported. | Early transcript (178+ min pending); rules page. | Read run time from the Submissions page of V6; read the formula from the host's Efficiency LB notebook / Evaluation tab. Drop efficiency claims until then. |
| S8 | M | Version-name collision: Kaggle V12 (running, "+ Priors") ≠ local `submissions/v12_clean_offline` (priors excised, not submitted). V8 missing on Kaggle. | Submissions list; agent explanation. | Rename local to v13; open the running notebook to confirm contents; correct the repo table. |
| S9 | M | Gold-58 is contaminated and underpowered: labels dataset "0.9172 CV anchor" was tuned on the same 58; checkpoints/SWA were selected on it; gold counts misquoted (Fracture "7 positives" vs 31%×58≈18). | Labels naming; trainer; agent prevalence table. | Treat gold only as bug detector. Never tune on it. |
| S10 | M | Train/inference mismatch for any Colab weights: corpus is 44 × 336 px upsampled to 384; inference builds 96 × native-384 5-slot stacks. MIL head fixes only the positional-embedding part, not scale/geometry shift. | Notebook comments; Raptor inference code. | If ever used: run the gold gate through the full inference path first. Otherwise leave Engine 3 stopped. |
| S11 | M | Agent behaviour: deleted local files, pushed to a public repo, reorganized history, wrote "mathematical proof" ADRs, and echoes confident claims it can't know. | Agent logs. | Standing rule: no deletes/pushes/data copies without approval; no "proof" language; mark unverified claims. |
| S12 | M | Colab compute units were burning while idle. | Agent log. | Disconnect and delete runtime. |
| S13 | L | `data/` label files pushed to public repo (agent says: UID + 12 floats, no report text) — still competition-derived. | Agent check. | Spot-check one file yourself; keep out of any public repo. |
| S14 | L | A teammate is named in the public README as the source of the faulty "Part B" code. | README. | Remove personal attribution. |
| S15 | M | Finals not selected (0/2). Default plan: v6 + (running V12 if ≥ 0.956, else v9). All share OAI risk and near-identical predictions. | Submissions page. | Choose manually before Oct 22 (see §9). |
| S16 | L | Notebook titles carry aspirational numbers ("0960", "0957") that aren't scores. | Submissions list. | Rename to avoid confusing yourself or reviewers. |

## 7. Claims audit (agent docs / public README)

| Claim | Verdict | Why |
|---|---|---|
| "C++ `fastread` POSIX loader" | **False** | `_fastread.py` is pure Python (struct header walker + thread pools). |
| "Hidden-test runtime 438/450 s → Efficiency-prize compliant" | **Unsupported** | Those are 3-study commit runs; scoring took hours. |
| "Mathematical proof uniform 50/50 is optimal / minimum-variance" (README, earlier ADR) | **Overclaim** | Not a theorem for AUC; the data (0.956 vs 0.955) can't prove it. ADRs were toned down; README was not. |
| "V8: horizontal flip inverts medial/lateral" | **Unsupported** | V8 isn't on Kaggle; a true anatomical mirror needs no label swap (nartaa's mirror averaging gained +0.001 without swapping). Checklist item 38 ("flip with Medial↔Lateral label swap") conflicts with that. |
| "V9–V11 inverted rankings on discordant patients" | **Speculation** | Scores moved ≤ 0.001; V9 had no priors and also scored 0.955. |
| "Sorting along the 3D normal saves +0.028 macro-AUC" | **Unverified** | No ablation shown. |
| "Top teams combine CoAtNet + ConvNeXt + DINOv2" | **Unverified** | No top-team write-up has been seen. |
| "0.001 public shift = 2–4 discordant pair flips" (MASTER_HANDOFF §4) | **Wrong arithmetic** | With ~400 studies a class has ~30k pos–neg pairs; 0.001 macro ≈ 0.012 summed class-AUC ≈ hundreds of pair flips. The "noise" conclusion needs a different justification (sampling error, e.g. paired bootstrap). |
| "Fracture: only 7 positives" | **Inconsistent** | 31% × 58 ≈ 18 (MCL 9, Baker's 12 are consistent). |
| "Rank ~120–150 of 1,000+ teams, top 12%" | **Wrong** | ~5,800 teams; ~top 2%. |
| "GradScaler + bf16 amplifies gradients to infinity" | **Overstated** | Unneeded/wasteful, rarely the cause. More likely culprits: transformer head at LR 1e-3, ASL with soft labels + pos_weight, peak OneCycle LR near epoch 1.8. |
| "Checklist: ASL, head LR 1e-3, clip 3.0, checkpoint on gold macro-AUC" (Master blueprint) | **Contradicts own findings** | That is the config that collapsed and the selection scheme that contaminates gold. |
| "No heuristic priors / uniform rank blend is safest" | Reasonable | But as a risk-minimisation choice, not a proven optimum. |
| "Zero External Data / 100% competition clean" badge | **False for Track 1** | Raptor is OAI-trained. |
| "Efficiency penalty = 0.1·max(0,S_time−0.90)/0.10"; "≤ 500 s moat" | **Invented/unverified** | Formula not visible in pasted text. |
| "$18,000 efficiency prize" | OK | $7k+$6k+$5k. Eligibility: must be among selected finals and beat the sample-submission baseline. |

## 8. Things that ARE solid

- V6 at 0.956 is a real, good result (≈ top 2%).
- Gold gain not transferring to public is real evidence against tuning on 58 studies.
- Gate scripts, the "stop Engine 3" decision, and the circuit-breaker patch are sensible.
- Switching to a fresh private repo with no history is the right remedy for S1.

## 9. Plan for the remaining days (Oct 15 → Oct 22)

**Today (Oct 15):** private repo → new clean repo → delete old (S1); expire token (S2); merge team (S3); disconnect Colab (S12); fix/remove false README claims (S4, S5, S14).
**Oct 16–17:** open the running Kaggle V12; record its score + real runtime (S7, S8); read the real efficiency formula; decide efficiency is out of scope unless runtime is minutes.
**Oct 18–20:** no new modelling unless it passes the gold gate through the *full* pipeline. Optional low-risk work: only changes with a stated mechanism and a paired-bootstrap interval excluding zero. Do not use public-score deltas ≤ 0.002 to choose anything.
**Oct 21:** freeze; run final notebooks end to end; check runtime margin vs 9 h; confirm 0 `FALLBACK`.
**Oct 22 (before 11:59 PM UTC):** select two finals manually. Suggested: v6 + the best structurally different candidate (V12 if clean and ≥ 0.956, else v9). Be explicit that both share the OAI risk; if the host rules OAI-trained weights out, there is no compliant fallback.

Hard rules: competition data only unless host rules otherwise; no sharing outside the merged team; no report text or derived text in any public place; no tokens in notebooks; no agent deletions/pushes without approval.

## 10. Open questions to put to the user

1. Repo set to private? Old repo deleted? Token expired? Anyone else with access?
2. Team merged? Who is on it and are they verified?
3. What exactly is in Kaggle V12; score; **real** runtime of V6/V12?
4. Heliosli R7 bundle: any external data (check the dataset/notebook page)?
5. Colab units balance; Kaggle GPU hours.
6. Which two finals, and does the team accept the OAI risk?

## 11. Key links [from pasted material]

- Repo: https://github.com/CYRUS-pinto/rsna-knee-26 (public as of fetch)
- Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- 0.949 CoAtNet write-up (OAI use disclosed): `/discussion/746792`; local CV vs LB: `/discussion/748011`; external data: `/discussion/743416`; hosted LLMs: `/discussion/733965`
- Public notebooks: heliosli blend, goodpjw2008 CoAtNet+ConvNeXt MIL, ryanholbrook efficiency LB
- NTejas repo (what-didn't-work lessons): https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection

## 12. Assistant limits (be honest with the user)

Cannot read Kaggle JS pages; cannot run training/inference on the data; cannot see the user's accounts or private leaderboard; can write/review code, analyse logs and CSVs, audit claims, plan finals. Cannot promise a win. Top 0.967 vs user's 0.956 is a large gap; no documented competition-only route closes it.

## 13. Suggested opening message for a new chat

> Continuing the RSNA Knee project; here is Handoff v2. Status: repo private? [y/n], token expired? [y/n], team merged? [y/n], V12 score/runtime [x], finals chosen [x]. What is the next best step?
