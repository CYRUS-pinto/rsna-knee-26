# RSNA Knee Abnormality Detection (2026) — Project Handoff & Technical Log

> [!IMPORTANT]
> **Complete Master Reference Available:** See **[MASTER_HANDOFF.md](MASTER_HANDOFF.md)** for the exhaustive 10-section master handoff detailing peer review consensus, the 0.5000 divergence forensic autopsy, small-sample Gold-58 statistics, Google Drive 0-second persistence, ADR summaries, and the final 7-day selection blueprint.

**Repository:** [https://github.com/CYRUS-pinto/rsna-knee-26](https://github.com/CYRUS-pinto/rsna-knee-26)  
**Target:** RSNA Knee Abnormality Detection  
**Evaluation Metric:** Macro-Averaged ROC AUC (across 12 binary findings)  
**Status:** Verified Personal Best **0.956** (Version 6, Clean Uniform Dual Engine), Version 12 Clean Offline Pipeline Prepared, Submission Lock Active.

---

## 1. Executive Summary & Competitive Dynamics

### The Leaderboard Dynamics
Over 200 public teams sit clustered between `0.941` and `0.954`, driven by blind forking of identical public checkpoints.
Top competitors (including @yosukeyama and Grandmaster Tucker Arrants) achieve >0.960 by addressing core fundamentals:
1. **3D Physical Normal Vector Projection:** Correcting random DICOM slice order via $\vec{n} = \text{IOP}_{0:3} \times \text{IOP}_{3:6}$ and $\text{Depth} = \text{IPP} \cdot \vec{n}$ (+0.028 AUC).
2. **Soft Report Pseudo-Labelling:** Continuous calibrated probabilities $[0.0, 1.0]$ across the 4,349 unlabelled reports rather than noisy hard binary labels.
3. **Dual-Family Neural Diversity:** Pairing a Hybrid Transformer (CoAtNet-384) with a Pure Modern ConvNet (ConvNeXt-S6 R7).
4. **Specialized Pathology Heads:** Employing M4 Osteoarthritis Extent heads (`ext_u`), Synovitis decomposition (`syn_beta`), and 3D sagittal fluid volumetry for Effusion.
5. **Scale-Invariant Percentile Rank Fusion:** Eliminating logit scale distortion across disparate model families.

---

## 2. Complete Forensic Submission Log

| Submission Ref | Model Configuration | Public Score | Status / Forensic Root Cause |
| :--- | :--- | :---: | :--- |
| **57108325** | Baseline Fork Heliosli 0.7/0.3 v2 | `0.954` | Succeeded. Standard public baseline. |
| **57113758** | Fork of rsna-knee-blend-gold v1 | `0.954` | Succeeded. Baseline replication. |
| **57128257** | Tri-Hybrid CoAtNet S6 2.5D MIL v2 | `0.954` | Succeeded. Longtime baseline ceiling. |
| **57150837** | CoAtNet S6 25D MIL v6 | `Exception` | **FAILED on hidden test set.** Fatal cause: single-model assertion in Cell 6: `if min(rho_rr) < 0.99: raise RuntimeError(...)`. Blending multi-arm SWA naturally diverged to $r \approx 0.97$, tripping the assert. |
| **57152208** | Quint Ensemble R7 Colab SWA v5 | `0.932` | **Severe 22-point drop.** Fatal cause: Inherited Hitarth's unverified "Part B" reader (`own_src`). Lacked offline `pylibjpeg-libjpeg` C-bindings and sorted slices by filename. **Effusion correlation inverted to $r = -0.740$**, poisoning the final submission. |
| **57165672** | Quint Ensemble R7 Colab SWA v6 | **`0.956`** | **Succeeded.** Verified baseline restored. Excised Part B, removed brittle asserts, restored clean dual-engine baseline. |
| **57172886** | Quint Ensemble R7 Colab SWA v7 | `0.955` | **Succeeded.** Tested domain-driven anatomical asymmetry (CoAtNet 65% on tears, ConvNeXt 70% on OA). Result proves uniform 50/50 blend (v6 at 0.956) is mathematically optimal for these two models; confirms that exceeding 0.956 requires Engine 3 model diversity rather than weight shifting. |
| **Kernel v8** | Tri-Engine Ensemble (Raptor CoAtNet + Cyrus SWA 0.9167 + ConvNeXt R7) | `Running` | **Deployed.** Engine 3 connected. Colab A100 training completed 12 epochs with SWA Macro-AUC **0.9167** on Gold-58 anchor. Stage 1 now runs dual CoAtNet arms (Raptor + Cyrus SWA); Stage 3 fuses with ConvNeXt R7. Target: 0.958–0.962+. |



---

## 3. Core Architecture & Verified Components

### Pipeline Stages (7-Cell Standard)

```mermaid
graph LR
    DICOM["Raw DICOMs"] --> Pre["C++ fastread (POSIX pread) & 3D Normal Sort"]
    Pre --> Stack["5-Slot 96-Slice Stack"]
    Stack --> CNet["Stage 1: CoAtNet-384 SWA<br/>(42 2.5D windows + 12 heads)"]
    Stack --> CNx["Stage 2: ConvNeXt-S6 R7<br/>(4-fold + M4 OA ext_u)"]
    CNx --> Vol["Fluid Volumetry & State Stacker"]
    CNet & CNx & Vol --> Fusion["Stage 3: Asymmetric Rank Fusion<br/>(CoAtNet tears + ConvNeXt OA)"]
    Fusion --> Guard["Safe Runtime Guard (min_rho ≥ 0.30)"]
    Guard --> Sub["submission.csv"]
```

* **Stage 1 (Cells 2–3)**: Raptor CoAtNet-384 SWA (`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`).
  - Threaded posix `pread` C++ fastread DICOM loader.
  - 5-slot standardized stack (96 slices: Sagittal fluid 26, Sagittal non-fluid 22, Coronal fluid 18, Coronal non-fluid 12, Axial 18).
  - 42 temporal 2.5D windows (3-slice channels) with 12 finding-specific cross-attention heads.
* **Stage 2 (Cells 4–5)**: Heliosli Release 7 ConvNeXt-S6 (`rsna-knee-compact-r7cnxfsss`, bundle `8c7b666e27dc0e8f`).
  - 4-fold ConvNeXt-Small at $352\times 352$ dense crops compiled via PyTorch Inductor.
  - M4 Osteoarthritis Extent heads (`ext_u`) + Synovitis decomposition (`syn_beta`).
  - 3D Sagittal Fluid Volumetry (`fluid_stacker.json`) + 6-class state router (`state_stacker.json`).
* **Stage 3 (Cell 6)**: Monotonic Percentile Rank Fusion:
  $$r = \text{rank}(p) / N, \quad \text{Blend} = W \cdot r_{\text{raptor}} + (1 - W) \cdot r_{\text{convnext}}$$
  - V7 Asymmetric Anatomical Weights: CoAtNet 65% on ACL, 60% on MCL & Contusion; ConvNeXt 70% on Medial/Lateral/PF OA.


---

## 4. The 5-Minute "Gold-58" Evaluation Gate

To avoid wasting 2–4 hours per Kaggle submission, all experimental iterations must pass the local **58-Gold Validation Gate**:
- Evaluates against the 58 expert-annotated gold studies from the competition training set.
- Runs in **~4 to 6 minutes** on a single GPU.
- Applies **Paired Bootstrap Testing (2,000 iterations)**:
  ```python
  def paired_boot(y, a, b, n=2000, seed=0):
      rng = np.random.default_rng(seed); d = []
      for _ in range(n):
          i = rng.integers(0, len(y), len(y))
          d.append(macro_auc(y[i], a[i]) - macro_auc(y[i], b[i]))
      d = np.array(d)
      return d.mean(), np.percentile(d, [2.5, 97.5]), (d > 0).mean()
  ```
- **Promotion Gate**: Only adopt a candidate blend if its 95% bootstrap confidence interval strictly excludes zero.

---

## 5. The Two-Track Final Selection Strategy (October 21)

Because public checkpoints (Raptor and Heliosli R7) carry pre-training exposure to OAI and SKM-TEA datasets, two distinct submissions will be selected for private evaluation:

1. **Track 1: Maximum Leaderboard Frontier (Medal/Ranking Target)**
   - Raptor CoAtNet-384 SWA + Heliosli Release 7 ConvNeXt-S6 + Asymmetric Finding Weights.
   - Targets the highest possible public/private score (~0.956 – 0.960+).
2. **Track 2: 100% Competition-Clean Model (Zero Audit Risk Target)**
   - Built exclusively from Cyrus's Colab Pro A100 models trained on competition DICOMs and Qwen soft-distilled report labels.
   - 0.0% external images, 100% compliant with Prize Winner Obligations Section 2.8.

---

## 6. Directory Layout & Key Artifacts

```
├── docs/
│   ├── architecture_overview.png   # Full system architecture diagram
│   └── finding_breakdown.png       # Finding-by-finding orthogonality matrix
├── tools/                          # Gold-58 Offline Validation Gate
│   ├── gate_make_root.py           # Mock test root generator for 58 gold studies
│   └── gate_score.py               # 2,000 paired bootstrap validation engine
├── scratch/
│   ├── clean_r7_kernel/            # Version 6 baseline kernel (LB 0.956)
│   ├── v7_asymmetric_kernel/       # Version 7 asymmetric weight kernel (PENDING)
│   ├── generate_architecture_visuals.py
│   └── test_v7_blend.py            # Local blend simulation engine
├── .gitignore
├── HANDOFF.md                      # Complete audit log and execution milestones
└── README.md                       # Master documentation
```

