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
- **Formula:**
  $$\text{Efficiency} = \frac{\text{AUC}}{\text{Benchmark} - \max\text{AUC}} + \frac{\text{RuntimeSeconds}}{32400}$$
- **Budget:** Keep total inference time under $500\text{ seconds}$ on Dual T4 GPUs.
  - Every $1,000\text{s}$ saved is equivalent to $+0.0143\text{ AUC}$ on the efficiency leaderboard.
  - Never sacrifice pipeline speed for marginal gain ($< 0.0005$ AUC).
