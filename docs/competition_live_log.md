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
