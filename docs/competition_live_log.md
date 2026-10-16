# Live Competition Log: RSNA 2026 Knee Abnormality Detection

### Active Candidate Status
- **V6 (0.956)**: [Ref `57165672`] - Slot 1 **PERMANENTLY LOCKED** (Benchmark Anchor).
- **V17 (MultiScale 384+224 + ConvNeXt R6 Flip TTA)**: [Ref `57253840`] - Status: **`COMPLETE` -> `0.958`** (443.4s on 3-study offline run; check Kaggle UI for hidden evaluation runtime). Top candidate for Slot 2.
- **V18 (Grandmaster MultiScale + Asymmetric Weights + Soft Synovitis Prior)**: [Ref `57260494`] - Status: `SubmissionStatus.PENDING` (Submitted `2026-10-16 07:48:55 UTC`). Kernel execution completed; scoring in progress.
- **V20 (Generalized Power-Rank Master Blend $p=1.4$)**: Kernel ID: `139004598` - Status: `KernelWorkerStatus.RUNNING` on Kaggle Dual Tesla T4 GPUs.

---

### Chronological Event Stream
- `[2026-10-16 03:54:09 UTC]` **V17 Kernel Submitted to Competition** (Ref: `57253840`). Offline benchmark verified at 443.4s on 3-study run.
- `[2026-10-16 07:15:30 UTC]` **V18 Engineering Complete**: Dual-arm Raptor (384+224) + ConvNeXt R6 + Anatomical Asymmetric finding weights + Soft Synovitis floor ($r_{synovitis} \ge 0.38 \cdot r_{effusion}$).
- `[2026-10-16 07:22:15 UTC]` Pushed `cyruspinto/rsna-knee-v18-grandmaster-blend` to Kaggle Dual T4 environment.
- `[2026-10-16 07:43:26 UTC]` V18 Kernel completed execution in 474.1s (3-study run). 0 fallbacks, 100% schema match.
- `[2026-10-16 07:48:55 UTC]` **V18 Officially Submitted to Competition** (Ref: `57260494`) via `competition_submit_code`.
- `[2026-10-16 08:10:00 UTC]` Verified submission queue: V17 (`57253840`) and V18 (`57260494`) undergoing Kaggle hidden test set evaluation (~1,300 studies).
- `[2026-10-16 08:53:08 UTC]` 🚀 **V17 SUBMISSION COMPLETE: 0.958 PUBLIC SCORE!** (Ref: `57253840`).
  - **Uplift**: Broke the 0.956 barrier (`0.956 -> 0.958`).
  - **Runtime**: 443.4s on the 3-study offline run; Kaggle hidden evaluation runtime to be verified in Kaggle Submissions UI.
  - **Role**: Current top candidate for **Slot 2**.
- `[2026-10-16 09:41:46 UTC]` **V18 Kernel Worker Completed** (`KernelWorkerStatus.COMPLETE`). Kaggle hidden metric evaluation in final scoring phase.
- `[2026-10-16 09:42:24 UTC]` 🚀 **Candidate V20 Pushed to Kaggle Dual T4 GPUs**: `cyruspinto/rsna-knee-v20-power-rank-blend` (Kernel ID: `139004598`). Status: `KernelWorkerStatus.RUNNING`. Generalized Power-Rank ($p=1.4$) continuous focal amplification.
- [2026-10-16 11:39:31 UTC] 🚀 **Candidate V20 Submitted & Kernel Session Completed** (KernelWorkerStatus.COMPLETE, Ref: 57267740). Kernel: cyruspinto/rsna-knee-v20-power-rank-blend. Kaggle hidden test evaluation in progress.
- [2026-10-16 11:45:00 UTC] **Engine 5 Multi-Plane Specialist Output Downloaded & Verified**:
  - Fold 0 OOF Macro-AUC: **0.79853** across all 12 classes (training/engine5_decoupled/kaggle_output/oof/oof_engine5_combined_fold0.parquet).
  - Key Specialists: Synovitis **0.8917**, Medial Meniscus **0.8625**, ACL **0.8563**, Contusion **0.8141**, Effusion **0.8138**.
  - Anti-collapse achieved: Mean off-diagonal correlation 0.4490 (vs >0.85 collapsed baseline).
- [2026-10-16 11:50:00 UTC] **Daily Submission Budget**: 3/5 used today (57253840 V17, 57260494 V18, 57267740 V20). 2 submissions remaining for today.
- [2026-10-16 12:05:00 UTC] **Full 5-Fold Benchmark (Arm A / CoAtNet-d4)**: Macro-AUC **0.85360** across all 4,407 unique studies in the dataset (tools/eval_arma_5fold.py). Medial Meniscus 0.9144, Synovitis 0.8883, Fracture 0.8818.
- [2026-10-16 12:07:00 UTC] **ADR-020 Formally Adopted**: Quad-Engine Multi-Plane Specialist Architecture & Full-Dataset Validation.
- [2026-10-16 12:09:19 UTC] 🚀 **Candidate V21 Pushed to Kaggle Dual T4 GPUs**: cyruspinto/rsna-knee-v21-quad-engine-blend. Status: RUNNING. Quad-Engine Multi-Plane Specialist Blend (Raptor-384, Raptor-224, ConvNeXt R6 3D joint crops, Engine 5 2.5D slice triples).
