# ADR-020: Quint-Engine Master Architecture, Cross-Slice Sequence MIL, and Anti-Queue-Trap Release Protocol

## Status
SUPERSEDED (Engine 6 failed at Gate 2 with offline AUC 0.52-0.54; weights and p=1.4 untested)

## Context
1. **Competition Timeline:**
   - Current Date: 2026-10-16.
   - Official Deadline: **2026-10-22 23:59:00 UTC** (~6 full days remaining).
   - Daily Submission Quota: 5 submissions per 24h rolling UTC window (4 used today, 1 remaining).
2. **The 0.958 Plateau Root Cause:**
   - Public 2D downsampled models plateau at `0.958` due to the sub-millimeter spatial resolution destruction when downsampling $512 \times 512$ native acquisitions down to $224 \times 224$.
   - Post-processing heuristics (such as V18's additive floors) violate ADR-002 and regress Wilcoxon rank-sum AUC to `0.957`.
3. **The Quint-Engine Master Solution:**
   - To break into $0.963\text{--}0.967+$ contention, five orthogonal anatomical engines are combined, each addressing distinct pathology profiles at native resolution:
     - **Engine 1 & 2 (Raptor-384 / Raptor-224 SWA):** Whole-volume macro context (CoAtNet-d4).
     - **Engine 3 (ConvNeXt R6 3D Crops):** SKM-TEA anchor bank joint crops with Flip TTA.
     - **Engine 4 / 5 (Engine 5 Axial Specialist):** 2.5D multi-plane fluid specialist (`Synovitis` 0.8917 AUC, `Effusion` 0.8138 AUC).
     - **Engine 6 (Native Joint Sequence MIL):** ConvNeXt-Small + 2-layer BiLSTM across 2.5D slice triplets $[c-1, c, c+1]$ preserving native 0.35 mm/pixel resolution for acute tears (`ACL`, `Medial Meniscus`, `Lateral Meniscus`, `Contusion`) and joint line degeneration (`MCL`, `Medial OA`, `Lateral OA`).

---

## Technical Specifications

### 1. Specialist Routing Matrix
Predictions $\hat{\mathbf{Y}}$ are formed via specialized domain weighting and generalized power-mean ($p=1.4$) on acute focal findings:

$$\hat{y}_{\text{focal}} = \left( w_r r^p + w_o o^p + w_5 e_5^p + w_6 e_6^p \right)^{1/p}$$

| Pathology Class | Primary Engine | Weights $(w_{\text{rap}}, w_{\text{ours}}, w_{e5}, w_{e6})$ | Power Exponent $p$ | Anatomical Justification |
| :--- | :--- | :--- | :--- | :--- |
| **ACL** | Engine 6 Sagittal | $(0.30, 0.30, 0.10, 0.30)$ | $1.4$ | Native 2.5D sagittal slice continuity captures ligament fiber disruption |
| **Medial Meniscus** | Engine 6 Sagittal | $(0.30, 0.30, 0.10, 0.30)$ | $1.4$ | Sub-millimeter resolution detects radial/horizontal cleavage tears |
| **Lateral Meniscus**| Engine 6 Sagittal | $(0.30, 0.30, 0.10, 0.30)$ | $1.4$ | Meniscal root tears require uncompressed pixel pitch |
| **MCL** | Engine 6 Coronal | $(0.30, 0.30, 0.10, 0.30)$ | $1.0$ | Coronal slice plane captures full femoral-tibial collateral span |
| **Medial OA** | Engine 6 Coronal | $(0.30, 0.30, 0.10, 0.30)$ | $1.0$ | Joint space narrowing visible on standing/coronal views |
| **Lateral OA** | Engine 6 Coronal | $(0.30, 0.30, 0.10, 0.30)$ | $1.0$ | Lateral joint compartment preserved without downsampling |
| **PF OA** | Raptor / Engine 5 | $(0.35, 0.35, 0.25, 0.05)$ | $1.0$ | Patellofemoral compartment best seen on axial fluid views |
| **Effusion** | Engine 5 Axial | $(0.35, 0.35, 0.25, 0.05)$ | $1.0$ | Suprapatellar pouch fluid volume in axial plane |
| **Synovitis** | Engine 5 Axial | $(0.35, 0.35, 0.25, 0.05)$ | $1.0$ | Synovial membrane thickening validated at 0.8917 AUC |
| **Baker's Cyst** | Raptor / Engine 5 | $(0.35, 0.35, 0.20, 0.10)$ | $1.0$ | Popliteal fossa fluid collection |
| **Contusion** | Engine 6 Sagittal | $(0.30, 0.30, 0.10, 0.30)$ | $1.0$ | Bone marrow edema patterns along lateral femoral condyle |
| **Fracture** | ConvNeXt / Engine 6| $(0.30, 0.30, 0.10, 0.30)$ | $1.4$ | Cortical breach requires acute high-confidence amplification |

---

## The "Anti-Queue-Trap" Operational Protocol

### 1. The Myth of the "Last 1-Hour" Submission
- **Common Amateur Mistake:** Waiting until 1 hour before competition deadline to submit final models.
- **The Failure Mode:** During the final 6 hours of any major Kaggle code competition, submission queues spike by 500% to 1000%. Hidden test evaluations that normally take 7 minutes end up queuing for 3 to 5 hours. Submissions initiated in the final hour almost always finish **after** the deadline passes, resulting in an un-scored submission and total disqualification.
- **The Top-1 Protocol:**
  - Complete, test, and lock candidate selections at least **12 to 24 hours prior** to official close.
  - Slot 1 (`0.956` baseline V6) is permanently locked as the conservative floor.
  - Slot 2 (`0.958` incumbent V17) will only be updated when an empirically superior checkpoint (`\ge 0.960`) has cleared the public leaderboard.

### 2. Daily Quota Allocation (Oct 16 to Oct 22)
- Today (Oct 16): 1 submission slot remaining. Reserved strictly for Candidate V22 (Quint-Engine Master Blend).
- Oct 17 – Oct 22: 5 slots/day $\times$ 6 days = 30 submission slots available to systematically explore:
  1. Full 5-fold out-of-fold calibrated ensembling.
  2. Multi-resolution dual-stream attention heads.
  3. Biomechanical relational graph heads.
