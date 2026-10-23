# Lessons Learned and Audit Notes

## 1. Failures that shaped the engineering rules *(from existing README)*

| Submission | Result | Root cause | Rule adopted |
|---|---|---|---|
| 57150837 | Exception | A hard `if min(rho) < 0.99: raise` check fired when a multi-arm blend legitimately lowered rank correlation, crashing the hidden-test run | Guardrails must not raise on the hidden set |
| 57152208 | 0.932 | An unverified alternative DICOM reader failed to load JPEG-lossless files and sorted by filename, which inverted Effusion (r = -0.74) | Use one verified reader; validate any change offline first |
| V7-V11 | 0.955 each | Asymmetric weights, flip TTA and heuristic priors each cost about 0.001 | Judge changes on the Gold-58 gate with a paired bootstrap, not on one leaderboard point |

V45's move from exceptions to warnings is consistent with the first rule.

## 2. Audit notes on the V20 and V45 notebooks

Checked by reading the code and saved outputs. These are accuracy notes for the write-up, not accusations about the models.

| Statement in a notebook header | What the notebook shows |
|---|---|
| V45: "Fail-soft Sovereign Rank Injection" is part of the architecture | No cell produces `sovereign_sub.csv`. The saved output prints that Sovereign predictions are absent, so the final run is the dual-arm anchor. The injection code never executed. |
| V45: "Zero Fatal Assertions" | Three assertions remain at the end of Stage 3 (UID order, shape `(n, 13)`, no NaN). Guardrail exceptions were converted to warnings, but not all assertions removed. |
| V45: rank standardisation is "mathematically optimal for ROC AUC" | Rank-based fusion is a sound choice for AUC and removes calibration differences. Optimality is not shown anywhere, and V20 uses the same ranks. |
| V20: "Zero hand-written floors, 100% smooth" | True of the blend formula. Fail-over rules (use the other half, fall back to 0.5) and hard `RuntimeError` guards are still present. |
| V20 / V45: per-finding weights "verified" | No validation output is stored in either notebook. The Gold-58 gate results behind these numbers should be added to the repo. |
| Raptor script docstring: "no ensembling and no test-time augmentation", 64 slices, 336 px, 0.924 LB | Stale text from the original recipe. The code runs 96 slices at 384 px, and the 384 model runs a plain plus mirrored pass. |

## 3. Open questions

1. **Coupling versus ADR-002.** The earlier project history found that heuristic cross-finding priors (V9-V11) regressed and removed them (V12, ADR-002). V45's ACL and Medial Meniscus boosts are the same kind of rule. The notebook contains no evidence that they help. Without a gate result, treat them as an unvalidated hypothesis and, if it matters, compare V20 and V45 on Gold-58 with the bootstrap rather than by leaderboard alone.
2. **Flip TTA.** V8 recorded flip TTA as a regression, yet both final candidates run flip TTA inside the Heliosli worker (MCL uses the plain pass, per the code comment) and mirror TTA in Raptor-384. The earlier explanation ("flip inverts medial and lateral anatomy") should be reconciled with this: either the V8 experiment implemented flipping differently, or the conclusion needs revisiting. The notebooks do not say which.
3. **Engine naming.** The earlier README describes the second engine as R7 ConvNeXt-S6; the final notebooks load the R6 frozen bundle. Confirm which was submitted for each version.
4. **Runtime margin.** Kaggle run times were 469.4 s (V20) and 533.9 s (V45), mostly from cold-start I/O in V45's run. Hidden-test timing depends on study count.
5. **Private versus public.** V20 scored 0.942 private, while V17 reached 0.958 public. They are different leaderboards, so this is not proof of a regression, but the gap between the two splits deserves a note when the V17 notebook is added.

## 4. What generalised

- A **single verified reader** with physically correct slice ordering was worth more than any blending trick (+0.028 AUC from ordering alone, per the earlier README).
- **Per-finding attention heads** let one model serve findings that live on different planes and sequences.
- **Rank fusion across architecturally different engines** was the reliable gain. Changing the *blend function* was not: V7 moved the score by about a thousandth (down). The effect of V20's and V45's blend changes is still to be recorded.
- A **3-study dry run is a smoke test only.** Correlation checks on 3 rows are meaningless, which is why V45 gates its diagnostics on `n > 20`.
