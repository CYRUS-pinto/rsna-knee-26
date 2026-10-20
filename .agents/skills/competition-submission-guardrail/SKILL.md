---
name: competition-submission-guardrail
description: Autonomous pre-flight validation and execution guardrail for Kaggle competition notebooks. Prevents fatal hidden-test-set assertions, Spearman sanity crashes, NaNs, schema mismatches, and timeout traps before submission.
---

# Competition Submission Guardrail (Automated Pre-Flight Gate)

## Purpose & Scope
This skill provides automated static analysis, mock scaling simulation, and mathematical verification for Kaggle notebooks before any commit or submission is executed. It completely eliminates "Notebook Threw Exception" failures caused by hidden-test scale mismatches, brittle correlation checks, and numerical instability.

## Mandatory Validation Checklist

Before any Kaggle kernel is pushed or submitted, execute the automated verification script:
`python tools/validate_kernel_guardrail.py <notebook_path>`

The guardrail validates five critical invariants:

### 1. Scale-Invariant Assertion Audit (The "N > 20" Trap)
- **Rule:** Never allow hard assertions or runtime exceptions on statistical distributions, rank correlations, or self-consistency checks that execute conditionally when test set size $N > 20$.
- **Forbidden:**
  - `if n > 20 and min(rho_rr.values()) < 0.99: raise RuntimeError(...)`
  - `assert rho > threshold`
  - `raise RuntimeError("... Spearman correlation mismatch ...")`
- **Enforcement:** Replace all such checks with non-fatal logger warnings (`warnings.warn(...)` or `print(...)`).

### 2. Numerical Stability & NaN Sanitization
- **Rule:** Half-precision (FP16) inference on GPUs (especially Tesla T4) can produce values exceeding $65,504$ during self-attention dot-products, resulting in `inf` or `NaN`.
- **Enforcement:**
  - Forbid `assert torch.isfinite(logits).all()` without prior sanitization.
  - Require float32 casting and clamping before metric evaluation:
    `logits = torch.nan_to_num(raw_logits.float(), nan=0.0, posinf=12.0, neginf=-12.0)`
  - Require final DataFrame sanity checks:
    `df[labels] = df[labels].fillna(0.5).clip(0.0001, 0.9999)`

### 3. Submission Format & Column Alignment
- **Rule:** Submission files must match `sample_submission.csv` identically in row IDs, column names, column order, and probability ranges.
- **Enforcement:**
  - `StudyInstanceUID` must be preserved as exact string matches.
  - Output shape must be $(N, 13)$ for the 12 target pathologies + ID column.
  - Predictions must be strictly in $[0, 1]$ (or $[0.0001, 0.9999]$ for log-loss stability).

### 4. Mock Large-Scale Execution Test ($N=1,500$)
- **Rule:** Kernels that pass with $N=3$ dummy test cases often fail when executed on the hidden test set ($N \approx 1,500$).
- **Enforcement:** Run the simulation harness with $N=1,500$ mock rows to verify:
  - Memory consumption remains within 13 GB RAM / 15 GB VRAM limits.
  - Multi-arm rank ensembling executes cleanly without index misalignment.
  - Fallback logic operates correctly when a model arm is missing or errors.

### 5. Efficiency Prize Optimization Gate
- **Official formula (see ADR-032; corrected 2026-10-20 — the previous version here was wrong):**
  $$E = \frac{\text{AUC} - M}{B - M} - \frac{T}{32400}$$
  $M$ = sample-submission baseline AUC, $B$ = best private AUC, $T$ = hidden-test runtime in seconds.
- **Trade-off:** 1,000 s of runtime costs $1000/32400 \approx 0.031$ of E, i.e. $\approx 0.031\,(B-M)$ AUC (≈ 0.014 AUC if $B-M \approx 0.47$). Verify $M$ before relying on this.
- **Eligibility:** only a submission that is one of the team's 2 selected finals (and beats the baseline) counts.
- **Runtime evidence:** use the runtime Kaggle reports for the hidden-test submission, never a 3-study interactive run.
- **Main-LB note:** the main leaderboard allows up to 9 h; do not cap the max-AUC candidate at 500 s.
