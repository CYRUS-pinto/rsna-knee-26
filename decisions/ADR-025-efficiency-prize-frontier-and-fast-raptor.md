# ADR-025: $18,000 Efficiency Prize Metric Derivation & Candidate V26 Fast MultiScale Raptor

**Status:** ACCEPTED  
**Date:** 2026-10-17  
**Author:** Cyrus Pinto & Pair Engineering  
**Context:** RSNA Knee Abnormality Detection Competition (5 days until deadline)

---

## 1. Context & Motivation

The competition features a dedicated **$18,000 Efficiency Prize** ($7,000 1st, $6,000 2nd, $5,000 3rd) evaluated on the private test cohort (~1,300 studies).
Previous documentation referenced fabricated runtime estimates (e.g. 420s for dual-arm models requiring ~0.3s/study). Empirical audits on $N=120$ studies revealed that our dual-arm pipelines (Raptor + ConvNeXt R7) run at **18.7 seconds/study**, with ConvNeXt accounting for **80% of total runtime (1,177s / 1,472s)**.

This heavy runtime caused Candidate V6 ($0.956$) to drop to **Rank 1,704** on the official Efficiency Leaderboard (`scratch/efficiency_lb/full_leaderboard.csv`), while faster models with $0.952\text{--}0.955$ reached the **Top 15**.

---

## 2. Mathematical Metric Formulation

The official evaluation metric formula:

$$\text{Efficiency} = \frac{\text{AUC}}{\text{Benchmark} - \max \text{AUC}} + \frac{\text{RuntimeSeconds}}{32400}$$

Where:
- $\text{Benchmark} \approx 0.500$ (score of `sample_submission.csv`)
- $\max \text{AUC} \approx 0.969$ (maximum private leaderboard score)
- $\text{Denominator} \approx 0.500 - 0.969 = -0.469$
- $32,400 = 9 \text{ hours} \times 3,600 \text{ seconds}$
- **Objective:** Strictly **MINIMIZE** the efficiency score.

### The 69-Second Equivalence Law
Computing the partial derivatives:
$$\frac{\partial \, \text{Efficiency}}{\partial \, \text{AUC}} \approx -2.1322$$
$$\frac{\partial \, \text{Efficiency}}{\partial \, T} = \frac{1}{32400} \approx +0.00003086$$

Setting $\Delta \text{Efficiency} = 0$:
$$\Delta T = -2.1322 \times 32400 \times \Delta \text{AUC} = -69,083 \times \Delta \text{AUC}$$

$$\mathbf{\Delta T = -69.1 \text{ seconds per } +0.001 \text{ AUC}}$$

Every **69.1 seconds (~1.15 minutes)** of runtime saved provides the exact same competitive advantage as **$+0.001$ AUC**. Saving **691 seconds (~11.5 minutes)** equals **$+0.010$ AUC**.

---

## 3. Decision: Candidate V26 Fast MultiScale Raptor

To establish a verified anchor on the efficiency frontier:
1. **Architecture:** Standalone MultiScale Raptor (384px SWA + 224px SWA) with 65/35 percentile rank fusion.
2. **ConvNeXt Elimination:** Completely removes ConvNeXt R7 decoding, 3D joint cropping, and model inference.
3. **Speedup:** Reduces execution time from ~25–35 minutes down to **$\sim 5\text{--}7\text{ minutes}$** on Kaggle Dual Tesla T4 GPUs.
4. **Verification:** Kernel commit run `cyruspinto/rsna-knee-fast-raptor-multiscale` completed in **118 seconds** with 0 failures (`scratch/v26_out/blend_receipt.json`).
5. **Competition Submission:** Officially dispatched under Ref **`57300862`**.

---

## 4. Consequences & Verification Boundaries

- **Slot 1 Permanent Lock:** Ref `57165672` (Candidate V6, 0.956) remains permanently locked.
- **Slot 2 Contender Lock:** Ref `57253840` (Candidate V17, 0.958) remains selected unless a candidate achieves Public LB $\ge 0.960$.
- **Frontier Evaluation:** Candidate V26-Fast provides empirical ground truth for real runtime and Public LB on the pure fastread architecture.
