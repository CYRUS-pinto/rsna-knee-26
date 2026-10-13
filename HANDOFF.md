# RSNA Knee Abnormality Detection (2026) — Complete Project Handoff & Technical Log

**Repository:** [https://github.com/CYRUS-pinto/rsna-knee-26](https://github.com/CYRUS-pinto/rsna-knee-26)  
**Target:** RSNA Knee Abnormality Detection  
**Evaluation Metric:** Macro-Averaged ROC AUC (across 12 binary findings)  
**Status:** Multi-Stage Model Architecture Built, 0.9172 Supervision Benchmark Verified, Colab Pro A100 Training Pipeline Deployed, Kaggle Evaluation Pipeline Operational.

---

## 1. Executive Summary & Competitive Dynamics

### The 0.954 Public Leaderboard Trap
Over 200 public teams sit clustered at exactly `0.941` to `0.954`. As confirmed by leading Grandmasters (DreadDevelopment, Mattia Angeli, Jiwei Liu), this congestion is driven by mass-forking of identical public checkpoints and micro-tweaking blend weights (e.g. `0.55/0.15` vs `0.60/0.10`). These weight modifications merely overfit the public holdout without producing genuine generalization.

### The Breakthrough Path
Top teams (e.g., #4 `azamat1ch`, #6 `diet1236364`) use zero external MRI data. Their advantage stems from:
1. **High-Fidelity Soft Report Pseudo-Labelling:** Generating calibrated continuous probabilities (0.0 to 1.0) across the 4,349 unlabelled reports rather than noisy hard binary labels.
2. **Dual-Family Neural Diversity:** Pairing a Hybrid Transformer (CoAtNet) with a Pure ConvNet (ConvNeXt-Small).
3. **Specialized Pathology Heads:** Employing RadImageNet ResNet-50 (pretrained on 1.3M radiology images) to recover signal on the three findings that bottleneck public models: **Patellofemoral Osteoarthritis (PF OA)**, **Lateral OA**, and **Synovitis**.
4. **Stochastic Weight Averaging (SWA):** Blending weights across top validation epochs for zero-inference-cost regularization.

---

## 2. Verified Supervision Benchmark: Super-Consensus Labels

The competition provides 4,407 knee MRI studies, but ground truth expert adjudication exists for only **58 studies** (the Gold set). The remaining 4,349 studies contain only unstructured multilingual radiology reports (Spanish, Dutch, German, French, English).

We engineered `SUPER_CONSENSUS_labels.csv` (and `.parquet`) across all 4,407 studies, blending multi-LLM extractions (Qwen2.5-32B, Qwen3-8B, DeepSeek) with finding-specific calibrated weighting.

### Benchmark Validation on the 58 Gold Studies:
* **Macro ROC AUC:** **`0.9172`** (Exceeds standard community band of 0.89–0.91)
* **Mean Brier Loss:** `0.1252`

| Finding | ROC AUC | Observed Gold Positive Rate | Model Mean Prediction | Bias |
|---|---|---|---|---|
| **ACL** | **1.0000** | 41.4% | 0.4791 | +0.065 |
| **MCL** | **0.9955** | 15.5% | 0.1748 | +0.020 |
| **Medial Meniscus** | **0.9615** | 44.8% | 0.5065 | +0.058 |
| **PF OA** | **0.9447** | 36.2% | 0.2286 | -0.133 |
| **Medial OA** | **0.9302** | 25.9% | 0.3913 | +0.133 |
| **Baker's** | **0.9203** | 20.7% | 0.3914 | +0.185 |
| **Fracture** | **0.9097** | 31.0% | 0.3995 | +0.089 |
| **Lateral Meniscus** | **0.8783** | 39.7% | 0.4192 | +0.023 |
| **Effusion** | **0.8770** | 60.3% | 0.7999 | +0.196 |
| **Contusion** | **0.8758** | 32.8% | 0.5250 | +0.197 |
| **Lateral OA** | **0.8704** | 19.0% | 0.3267 | +0.137 |
| **Synovitis** | **0.8435** | 46.6% | 0.4380 | -0.028 |

### Monotonic Probability Calibration Curve:
* Bins `[0.0, 0.1)`: Observed positive rate = **2.4%**
* Bins `[0.1, 0.2)`: Observed positive rate = **13.2%**
* Bins `[0.3, 0.4)`: Observed positive rate = **34.6%**
* Bins `[0.4, 0.5)`: Observed positive rate = **53.8%**
* Bins `[0.8, 0.9)`: Observed positive rate = **64.0%**
* Bins `[0.9, 1.0)`: Observed positive rate = **87.0%**

---

## 3. Key Technical Discoveries & Audits

1. **The "Apex V21" Placebo Audit:**
   * Public kernels claiming "Quad-Backbone" (e.g. `rsna-knee-apex-grandmaster-stack-v21`) defined a placeholder blend function `blend_quad_backbone_submission(parent_path, rad_path, out_dir)`, but **never invoked it** and never generated `rad_path`. Forks were unknowingly running the standard tri-blend.
2. **True RadImageNet Pipeline Located & Verified:**
   * Found Grandmaster Jiwei Liu's genuine working implementation (`rsna-knee-fast-2xt4-inference.ipynb`, Cells 24–25).
   * It uses ResNet-50 pretrained on RadImageNet with multihead attention (`_RadHead`) over plane tokens (Sagittal FS, Coronal FS, Axial FS) and slice position tokens, with a fitted linear calibrator payload for protocol metadata.
3. **Compartmental Decoupling in Ensembling:**
   * Empirical testing confirmed that cross-bleeding Medial OA predictions into Lateral OA causes sharp AUC collapse (`0.8704 → 0.8298`). Medial and lateral knee compartments are anatomically and mechanically independent (varus vs valgus wear); their feature extractors must remain decoupled.
4. **Ultra-Fast DICOM Pipeline (`_fastread.py`):**
   * Integrates 22 custom I/O reading paths (`study_lut`, `hybrid_lut`, `onepass_fast`), using custom VR walkers, single-pass memory mapping, and lookup-table histogram percentiles. Benchmarked at **3 studies in 3 seconds** (10x faster than standard pydicom).

---

## 4. Cloud Training & Submission Workflow

### A. Google Colab Pro Training (High-Speed Engine)
* **Notebook:** [`colab_rsna_knee_super_train.ipynb`](./colab_rsna_knee_super_train.ipynb)
* **Hardware:** Nvidia A100 (40GB/80GB) or L4 GPU.
* **Speed:** ~2–3 minutes per epoch (entire 12-epoch run finishes in ~25–35 minutes, compared to 2.5 hours on Kaggle).
* **Workflow:**
  1. Open Colab → Upload `colab_rsna_knee_super_train.ipynb` → Select A100 GPU.
  2. Runs end-to-end data download (~90s via Kaggle API), trains CoAtNet RMLP 2 RW 384 with TF32/BF16 autocast.
  3. Evaluates against Gold 58 every epoch, generates SWA weights (`colab_super_coatnet_0917_swa.pt`).
  4. Automatically packages and creates private Kaggle dataset `cyruspinto/rsna-knee-colab-super-weights` via the CLI.

### B. Kaggle Inference & Evaluation Pipeline
* **Kaggle Kernel:** `cyruspinto/rsna-knee-coatnet-s6-25d-mil-0957`
* **Status:** Dual-GPU workers confirmed `PASS` on both Tesla T4s (0 fallbacks, 158s run).
* **Submission Log:**
  * `Ref 57108325`: Baseline Heliosli 0.7/0.3 → **0.954**
  * `Ref 57113758`: Restored Gold Blend → **0.954**
  * `Ref 57128257`: Tri-Hybrid CoAtNet S6 2.5D MIL 0957 v2 → In final evaluation scoring queue.

---

## 5. Repository File Map

```
├── colab_rsna_knee_super_train.ipynb       # Turnkey Colab Pro A100 training notebook with SWA & Kaggle export
├── SUPER_CONSENSUS_labels.csv              # Calibrated soft labels (4,407 studies, 0.9172 Gold AUC)
├── SUPER_CONSENSUS_labels.parquet          # Compressed parquet format for fast cloud loading
├── _fastread.py                            # High-speed C-speed DICOM reader engine (10x speedup)
├── rsna_knee_submission_v2.ipynb           # Validated Kaggle Dual-T4 Tri-Hybrid inference kernel
├── scratch/
│   ├── evaluate_calibration.py             # Gold 58 calibration & Brier loss evaluator
│   ├── test_calibration_tuning.py          # Anatomical cross-finding prior testing
│   ├── winners_report.md                   # Complete forum intelligence & GM architecture breakdown
│   ├── rad_module.py                       # Extracted & validated RadImageNet ResNet-50 calibrated heads
│   ├── quad_kernel/                        # Staged 4-backbone inference kernel with RadImageNet
│   ├── eval_coatnet_output/                # Downloaded receipts, timing metrics & worker logs
│   └── kernels/                            # Pulled Grandmaster reference implementations
```

---

## 6. Immediate Next Steps

1. **Launch Training on Colab Pro:**
   * Open `colab_rsna_knee_super_train.ipynb` on Colab with A100 to generate your proprietary `colab_super_coatnet_0917_swa.pt` weights.
   * Optionally run a second pass with `--arch convnext_small.fb_in22k_ft_in1k_384` to create a companion ConvNeXt checkpoint.
2. **Fuse Weights into Inference Stack:**
   * Attach `cyruspinto/rsna-knee-colab-super-weights` to the staged Quad-Backbone inference kernel (`scratch/quad_kernel/`).
   * Submit to Kaggle to challenge for the top medal tier (target: **0.958 – 0.966+**).
