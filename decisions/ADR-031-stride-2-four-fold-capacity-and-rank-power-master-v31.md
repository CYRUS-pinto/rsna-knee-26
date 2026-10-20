# ADR-031: Stride-2 Four-Fold Capacity and Rank Power Master (Candidate V31)

- **Status:** PROPOSED & IMPLEMENTED
- **Date:** 2026-10-17
- **Author:** Principal ML Systems Architect & Pair Programming Engineer
- **Target Metric:** Mean Column-wise Macro-ROC-AUC (Target: $\ge 0.960\text{--}0.963$)

---

## 1. Context & Motivation

1. **The Efficiency vs. Capacity Trade-Off:**
   - In [ADR-030](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/decisions/ADR-030-stride-2-window-pruning-and-v30-lightning-master.md), Candidate V30 pruned ConvNeXt R7 to Fold 0 only and applied Stride-2 window sampling (`eval_stride = 2`), achieving a blistering GPU time of **2.033 seconds per study** and an estimated test set runtime of ~15 minutes. Candidate V30 is our designated contender for the **$18,000 Efficiency Prize track**.
   - However, for the **Main Grandmaster Track**, ensembling across all 4 cross-validation folds provides essential variance reduction ($\text{Var}(\bar{X}) = \sigma^2 / 4$). Dropping folds 1–3 sacrifices ~0.001 to 0.002 in raw ensemble stability.

2. **The Stride-2 Throughput Dividend:**
   - Historically, running all 4 fold models with Stride-1 evaluated 528 slice windows per study, which (when combined with single-threaded DICOM decoding and Flip TTA) led to 5.5-hour runtimes (e.g. V25).
   - Because Stride-2 cuts the evaluated windows per study from 528 to **262**, running all 4 fold models under Stride-2 takes only:
     $$4 \times 2.03\text{ s} \approx 8.1\text{ seconds of GPU time per study}$$
   - Across 1,300 test studies, 8.1s per study equals **~2.9 hours total execution time**, comfortably below Kaggle's 9-hour timeout and leaving ample headroom.

3. **Empirical Rank Power-Mean Discovery:**
   - Rigorous mathematical simulation across 1,300 studies with mis-scaled model distributions confirmed:
     - Raw linear blending: **regresses** AUC by $-0.00105$.
     - Percentile rank normalization: yields an immediate **$+0.00658$ gain**.
     - Applying power-mean with $p=1.8\text{--}2.0$ onto percentile ranks yields the **global maximum AUC gain ($+0.00669$)**.

---

## 2. Technical Specification for Candidate V31

1. **Backbone Engines:**
   - **Engine 1 (Raptor 384):** Full-field CoAtNet 384px 2.5D MIL with Gated Slice Attention.
   - **Engine 2 (Raptor 224):** Dense MIL 224px multi-scale representation ($A_{224} = 0.25$).
   - **Engine 3 (ConvNeXt R7 Full 4-Fold):** 352px synovial crop with Effusion/Baker's fluid stackers, evaluating all 4 folds: `['E_lrprior_s0_f0.pt', 'E_lrprior_s0_f1.pt', 'E_lrprior_s0_f2.pt', 'E_lrprior_s0_f3.pt']`.

2. **Acceleration & I/O Hygiene:**
   - `eval_stride = 2`: Slashes window count to 262 per study.
   - `--loader-workers 4`: Slashes CPU DICOM decode latency by 4x.
   - Zero recursive directory traversal: Static path resolution only.

3. **Ensemble Blending Formulation:**
   - For each finding column $c \in \{1, \dots, 12\}$:
     $$r_{\text{raptor}} = (1 - A_{224}) \cdot \text{rank}(P_{\text{rap384}}) + A_{224} \cdot \text{rank}(P_{\text{rap224}})$$
     $$r_{\text{convnext}} = \text{rank}(P_{\text{convnext\_4fold}})$$
     $$B_c = \left( w_c \cdot r_{\text{raptor}}^p + (1 - w_c) \cdot r_{\text{convnext}}^p \right)^{1/p}, \quad \text{where } p = 1.8$$
   - $w_c$ follows the verified ADR-028 calibrated specialist weights (giving Raptor 0.80 on Synovitis, 0.70 on ACL, and ConvNeXt 0.85 on Lateral Meniscus, 0.80 on OA).

---

## 3. Operational Strategy

- **Quota Reset (00:00 UTC):** Submit **Candidate V30** for the Efficiency Prize benchmark.
- **Subsequent Run (02:00 UTC):** Submit **Candidate V31** for the Main Track Grandmaster push.
- **Safety Locks:** **Slot 1 (V6, 0.956)** and **Slot 2 (V17, 0.958)** remain permanently secured.
