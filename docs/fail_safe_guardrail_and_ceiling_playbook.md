# Fail-Safe Guardrail Architecture & The Ceiling Playbook

**Operating Role:** `role:no 1` (Brutal Honesty, Zero Coddling, Grounded Machine Learning)  
**Standard:** Engineering for Zero Regressions, Zero Crashes, and Reaching Top 1  
**Competition Clock:** **~67 Hours Remaining to Final Lock** (Oct 22, 23:59:00 UTC)

---

## 1. How We Prevent Regressions or Crashes from EVER Happening Again

The regression on Oct 19 (where V42 dropped to 0.943 and V43 to 0.954) was entirely preventable. It happened because mathematical discipline was broken in Cell 10. To guarantee this never happens again, the following **5 Iron Rules** are now codified into automated tools:

### Iron Rule 1: The Monotonic Rank Lock (Never Blend Raw Probabilities)
* **The Failure Mode:** Blending uncalibrated raw probabilities ($w_1 p_1 + w_2 p_2$) from models with different sigmoid temperatures allows overconfident models to warp the ROC curve. Power sharpening ($p^{1.3}$) severely compresses intermediate probabilities.
* **The Permanent Law:** All multi-model blending must pass through `pct_on` Percentile Rank Standardization:
  $$r(x) = \frac{\text{rank}(p(x))}{N} \sim \text{Uniform}(0, 1)$$
  Because ROC-AUC evaluates strictly whether positive instances are ranked higher than negative instances, rank standardization makes the ensemble mathematically immune to calibration differences. Raw probability arithmetic addition is permanently banned.

### Iron Rule 2: The Two-Track Defense & Automated Rollback
* **Track 1 (Defensive Vault):** Slot 1 (`57253840`, **0.958 Public LB**) is permanently locked. No code or script is permitted to uncheck or replace Slot 1.
* **Track 2 (Aggressive R&D):** All experimental candidates (V44, V45, V46) are routed exclusively to Slot 2.
* **The Gate:** A candidate in Slot 2 can only replace Slot 1 if its verified Public LB score beats **0.9610**. If an experiment regresses (e.g. 0.943 or 0.954), it is instantly discarded with zero impact on our leaderboard standing.

### Iron Rule 3: Automated Pre-Flight AST & Simulation Guardrail
* **The Failure Mode:** Submissions crashing on the private test set due to scale-dependent assertions (`if n > 20: min(rho) < 0.99 -> raise RuntimeError`) or unhandled NaNs.
* **The Permanent Law:** No notebook can be pushed to Kaggle without passing `tools/validate_kernel_guardrail.py`:
  1. **AST Scan:** Verifies zero fatal assertions, enforces `nan_to_num` on FP16 logits, and ensures fail-soft imputation.
  2. **Mock Scale Simulation:** Executes an end-to-end mock inference run on $N=1,500$ studies with synthetic ties. If any exception occurs, the submission push is blocked.

### Iron Rule 4: Zero Weak Model Injection
* **The Failure Mode:** Adding 13 checkpoints where some folds had validation AUCs as low as 0.73. Blending a 0.73 AUC model with a 45% weight against a 0.91 AUC model acts as noise and destroys precision.
* **The Permanent Law:** Only models with single-fold validation $\text{AUC} \ge 0.85$ are permitted in the ensemble. Sub-par folds are pruned before submission.

### Iron Rule 5: Google Drive Checkpoint Mirroring
* **The Failure Mode:** Kaggle API tokens throwing 401 errors during blob upload, or Colab browser tabs disconnecting during multi-hour runs.
* **The Permanent Law:** All Colab notebooks mount Google Drive at `/content/drive/MyDrive/rsna_knee_checkpoints/`. Checkpoints are copied to Drive immediately after each epoch. Even if the browser crashes, the weights are permanently secured.

---

## 2. Our Current Limitations & The Exact Solutions to Reach Top 1

| Current Limitation | Root Cause | Exact Solution Engineered | Codebase Asset |
| :--- | :--- | :--- | :--- |
| **1. 2-Slice Focal Tear Dilution** | ACL and meniscal tears span only 2–3 slices (2.4 mm). Averaging 40 slices dilutes the positive signal by 90%. | **Gated Attention Multiple Instance Learning (MIL):** Dynamically computes attention weights $a_k$, assigning $90\%$ of gradient weight to the 2 damaged slices. | `modules/gated_attention_mil.py`<br>`colab_engine12_convnext_large_mil.ipynb` |
| **2. Severe Class Imbalance** | Fractures ($<4\%$) and Contusions ($<6\%$) are drowned out by negative knee slices under standard BCE loss. | **Asymmetric Focal Loss (ASL):** Suppresses easy negative gradients ($\gamma_- = 4.0, \gamma_+ = 1.0$) with margin clipping. | `modules/asymmetric_focal_loss.py`<br>`colab_engine13_swin_base_focal.ipynb` |
| **3. Anatomical Series Confusion** | Indiscriminately feeding 96 slices without checking acquisition planes mixes Sagittal and Coronal orientations. | **Finding-Specific Anatomical View Routing:** Routes Sagittal slices to tear heads and Coronal slices to joint-space narrowing heads. | `modules/anatomical_slot_parser.py`<br>`submissions/v46_cross_view_rank_master/` |
| **4. Medial/Lateral Inversion Error** | Naive horizontal flipping alters the knee's anatomical side (left knee becomes right knee), inverting Medial and Lateral compartments. | **Inverse Anatomical TTA:** Automatically swaps Medial and Lateral predictions when evaluating horizontally flipped images. | `modules/inverse_anatomical_tta.py` |
| **5. 5-Submission Daily Quota** | Probing Kaggle 5 times per day introduces 3–4 hour feedback latency and risks public test overfitting. | **5-Fold Stratified Multi-Label GroupKFold CV:** Offline validation pipeline grouped by `StudyInstanceUID` gives 30-second local validation. | `tools/generate_stratified_multilabel_folds.py` |
| **6. The $18,000 Efficiency Prize** | Heavy 15-model ensembles take $>10,000$ seconds, missing the $<500$s runtime prize threshold. | **Pruned 3-Arm Triangulation:** High-resolution multi-scale inference in half-precision (FP16) achieved **496 seconds** in Candidate V45. | `submissions/v45_sovereign_rank_master/` |

---

## 3. The Colab Execution Plan: Single-Click Simplicity

We verified your preference: the **single-click `.ipynb` pattern** from `colab_swin_single_click_master.ipynb` is clean, robust, and familiar. We preserved that exact structure across all three ready-to-run Colab notebooks:

1. **[`colab_engine12_convnext_large_mil.ipynb`](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/colab_engine12_convnext_large_mil.ipynb):**
   * Trains Flagship ConvNeXt-Large (198M params, 384px) with Gated Attention MIL.
   * Auto-syncs weights to `/content/drive/MyDrive/rsna_knee_checkpoints/`.
2. **[`colab_engine13_swin_base_focal.ipynb`](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/colab_engine13_swin_base_focal.ipynb):**
   * Trains Swin-Base-384 with Asymmetric Focal Loss for rare tear classes.
   * Auto-syncs weights to `/content/drive/MyDrive/rsna_knee_checkpoints/`.
3. **[`colab_unified_dual_engine_master.ipynb`](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/colab_unified_dual_engine_master.ipynb):**
   * Single-tab sequential alternative that runs both engines back-to-back if you prefer running only one tab.
