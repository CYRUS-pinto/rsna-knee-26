# Live Competition Log: RSNA 2026 Knee Abnormality Detection

### Active Candidate Status
- **V6 (0.956)**: [Ref `57165672`] - Slot 1 **PERMANENTLY LOCKED** (Benchmark Anchor).
- **V17 (MultiScale 384+224 + ConvNeXt R6 Flip TTA)**: [Ref `57253840`] - Status: `SubmissionStatus.PENDING` (Submitted `2026-10-16 03:54:09 UTC`).
- **V18 (Grandmaster MultiScale + Asymmetric Weights + Soft Synovitis Prior)**: [Ref `57260494`] - Status: `SubmissionStatus.PENDING` (Submitted `2026-10-16 07:48:55 UTC`).

---

### Chronological Event Stream
- `[2026-10-16 03:54:09 UTC]` **V17 Kernel Submitted to Competition** (Ref: `57253840`). Offline benchmark verified at 443.4s ($18k Efficiency Eligible).
- `[2026-10-16 07:15:30 UTC]` **V18 Engineering Complete**: Dual-arm Raptor (384+224) + ConvNeXt R6 + Anatomical Asymmetric finding weights + Soft Synovitis floor ($r_{synovitis} \ge 0.38 \cdot r_{effusion}$).
- `[2026-10-16 07:22:15 UTC]` Pushed `cyruspinto/rsna-knee-v18-grandmaster-blend` to Kaggle Dual T4 environment.
- `[2026-10-16 07:43:26 UTC]` V18 Kernel completed execution in 474.1s. 0 fallbacks, 100% schema match.
- `[2026-10-16 07:48:55 UTC]` **V18 Officially Submitted to Competition** (Ref: `57260494`) via `competition_submit_code`.
- `[2026-10-16 08:10:00 UTC]` Verified submission queue: V17 (`57253840`) and V18 (`57260494`) undergoing Kaggle hidden test set evaluation (~1,300 studies).
- `[2026-10-16 08:10:13 UTC]` Competition Monitor Daemon started (tracking V17 [57253840] and V18 [57260494]).
