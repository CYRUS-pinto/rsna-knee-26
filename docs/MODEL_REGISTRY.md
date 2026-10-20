# Model Registry — The Definitive Asset Truth Table

**Updated:** 2026-10-20 16:25 UTC  
**Source of Truth:** Verified directly against Kaggle Datasets API (`cyruspinto`) and repository checkpoints.

> [!IMPORTANT]
> **MANDATORY CHECK FOR ALL AGENTS:**  
> Before proposing any new training run, check this table. Do NOT re-train existing completed folds. Do NOT propose redundant blends of same-family models.

---

## 1. Verified Kaggle Dataset Assets (Ready for Inference)

| Engine | Architecture | Input Res | Folds Completed | Kaggle Dataset Slug | File Size | Verified Quality / Score | Status |
|---|---|---|---|---|---|---|---|
| **Raptor** | CoAtNet multi-scale | 384+224 mixed | 5-Fold (Public) | Public Notebook Anchor | Internal | ~0.87 Macro AUC | **Anchor Arm (70% Blend)** |
| **ConvNeXt R6** | ConvNeXt-Small | 384 | 4-Fold (Public) | Public Notebook Anchor | Internal | ~0.85 Macro AUC | **Available** |
| **Super-CoAtNet** | CoAtNet-384 | 384 | 1-Fold (Public) | Public Notebook Anchor | Internal | 0.9139 Gold-58 AUC | **Included in V48** |
| **Engine 10** | Swin-Base-384 | 384 | **ALL 5 FOLDS (0, 1, 2, 3, 4)** | `cyruspinto/rsna-knee-engine10-fold{0..4}-weights` | ~323 MB each | Loss ~0.4078 | **Completed on Kaggle** |
| **Engine 11** | ConvNeXt-Base-384 | 384 | **ALL 5 FOLDS (0, 1, 2, 3, 4)** | `cyruspinto/rsna-knee-engine11-fold{0..4}-weights` | ~326 MB each | 0.8712 Gold-58 AUC | **Included in V48** |
| **Engine 9** | ConvNeXt-Small 2.5D | 384 | **3 FOLDS (0, 1, 2)** | `cyruspinto/rsna-knee-engine9-fold{0..2}-weights` | ~185 MB each | 0.8325 Gold-58 AUC | **Available** |
| **Engine 12** | ConvNeXt-Large-384 Gated MIL | 384 | **Fold 0 (DONE)** | `cyruspinto/rsna-knee-engine12-convnext-large-weights` | **788.3 MB** | **0.8808 Peak Gold-58 AUC** | **Flagship Arm (Active in V48)** |
| **Engine 13** | Swin-Base-384 Focal MIL | 384 | **Fold 0 (DONE)** | `cyruspinto/rsna-knee-engine13-swin-base-weights` | **348.8 MB** | **0.8577 Peak Gold-58 AUC** | **Flagship Arm (Active in V48)** |

---

## 2. Empty Placeholder Stubs (Need Training Folds)

These datasets exist on Kaggle as dataset containers, but currently contain only `placeholder.txt` (~36 bytes) and **MUST be trained and uploaded**:

| Planned Engine | Architecture | Target Fold | Target Dataset Slug | Notebook to Run | Required GPU | Priority |
|---|---|---|---|---|---|---|
| **Engine 12** | ConvNeXt-Large-384 Gated MIL | **Fold 1** | `cyruspinto/rsna-knee-engine12-fold1-weights` | `colab_engine12_fold1_master.ipynb` | **A100** | **CRITICAL (Priority 1)** |
| **Engine 13** | Swin-Base-384 Focal MIL | **Fold 1** | `cyruspinto/rsna-knee-engine13-fold1-weights` | `colab_engine13_fold1_master.ipynb` | **T4 / L4** | **CRITICAL (Priority 1)** |
| **Engine 12** | ConvNeXt-Large-384 Gated MIL | **Fold 2** | `cyruspinto/rsna-knee-engine12-fold2-weights` | Clone Fold 1 with `fold=2` | **A100** | Priority 2 |
| **Engine 13** | Swin-Base-384 Focal MIL | **Fold 2** | `cyruspinto/rsna-knee-engine13-fold2-weights` | Clone Fold 1 with `fold=2` | **T4 / L4** | Priority 2 |

---

## 3. Active Leaderboard & Submission Telemetry

- **Permanent Safety Anchor:** Slot 1 (Ref `57253840`, **0.958 Public LB**) — **PERMANENTLY LOCKED**.
- **Candidate V48 (`57443706`):** Apex Quad-Arm Sovereign Ensemble (Raptor-384 anchor + Engine 12 ConvNeXt-Large 0.8808 + Engine 13 Swin-Base 0.8577 + Engine 11).
  - Status: **`PENDING`** on Kaggle hidden test scoring backend.
  - Tracked autonomously by local daemon `tools/autonomous_overnight_monitor.py` (PID task ID `task-154`).
- **Daily Quota Remaining:** **1 out of 5 submissions available** today (resets at 00:00 UTC / 05:30 IST).
- **Time Remaining to Final Cutoff:** **55.9 hours** (Deadline: 2026-10-22 23:59 UTC).

---

## 4. Immediate Operating Instructions for Any AI Agent

1. **DO NOT run `colab_swin_single_click_master.ipynb` for Engine 10 Folds 2/3:**  
   Those weights already exist on Kaggle. Re-running it will waste compute time.
2. **DO run `colab_engine13_fold1_master.ipynb` on Colab Tab 1 (T4/L4):**  
   Trains Engine 13 (Swin-Base-384 Focal MIL) Fold 1 and uploads weights to `cyruspinto/rsna-knee-engine13-fold1-weights`.
3. **DO run `colab_engine12_fold1_master.ipynb` on Colab Tab 2 (A100):**  
   Trains Engine 12 (ConvNeXt-Large-384 Gated MIL) Fold 1 and uploads weights to `cyruspinto/rsna-knee-engine12-fold1-weights`.
4. **DO monitor Candidate V48:**  
   Check [docs/LIVE_STATE.md](LIVE_STATE.md) for the score. Once V48 scores, combine Fold 0 + Fold 1 weights into Candidate V49 for an out-of-fold calibrated boost toward 0.965+.
