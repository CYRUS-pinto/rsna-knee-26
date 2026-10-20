# RSNA Knee Abnormality Detection: Candidate Efficiency Table

**Status:** PENDING KAGGLE UI RUN TIME ENTRIES  
**Evidence Standard:** MEASURED (API), ESTIMATED (Formula), CLAIMED (Unverified)  

---

## 1. Candidate Submissions Summary

| Ref | Version | Public Score | Kaggle Run Time (Hidden Test) | $E = \frac{\text{AUC}}{B - M} + \frac{T}{32,400}$ | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **57165672** | Candidate V6 | **0.956** (MEASURED) | `PENDING USER PASTE` | `PENDING RUN TIME` | COMPLETE | Offline Clean Anchor (Slot 1 Lock) |
| **57253840** | Candidate V17 | **0.958** (MEASURED) | `PENDING USER PASTE` | `PENDING RUN TIME` | COMPLETE | MultiScale (384+224) + ConvNeXt R6 |
| **57260494** | Candidate V18 | **0.957** (MEASURED) | `PENDING USER PASTE` | `PENDING RUN TIME` | COMPLETE | Heuristic Clamping Floor |
| **57267740** | Candidate V20 | **0.958** (MEASURED) | `PENDING USER PASTE` | `PENDING RUN TIME` | COMPLETE | Power-Rank Focal $p=1.4$ |
| **57273256** | Candidate V21 | **0.949** (MEASURED) | `PENDING USER PASTE` | `PENDING RUN TIME` | COMPLETE | Quad-Engine Blend (Collapsed) |
| **57280117** | Candidate V22 | **0.957** (MEASURED) | `PENDING USER PASTE` | `PENDING RUN TIME` | COMPLETE | Decoupled Compartmental Blend |
| **57293215** | Candidate V23 | **0.957** (MEASURED) | `PENDING USER PASTE` | `PENDING RUN TIME` | COMPLETE | 384px R7 MultiScale + Raptor Dual-Crop |
| **57298964** | Candidate V24 | `PENDING SCORING` | `SCORING ON HIDDEN TEST` | `PENDING SCORE` | PENDING | ADR-024 Uniform Master Blend |
| **57300862** | Candidate V26-Fast | **0.949** (MEASURED) | `PENDING USER PASTE` | `PENDING RUN TIME` | COMPLETE | Pure Fastread MultiScale Raptor (Efficiency Frontier Baseline) |
| **57301656** | Candidate V25 | `PENDING SCORING` | `SCORING ON HIDDEN TEST` | `PENDING SCORE` | PENDING | 384px R7 MultiScale + Raptor Dual-Crop + ADR-024 Uniform + Flip TTA |

---

## 2. Break-Even Formula & Sensitivity

Assuming:
* $M$: Maximum public AUC achievable (approx. $0.967$ on public leaderboard)
* $B$: Baseline benchmark AUC
* $\Delta(M - B) \approx 0.467$

$$\text{Break-Even Seconds per } 0.001 \text{ AUC} = \frac{32,400 \times 0.001}{M - B} = \frac{32.4}{M - B}$$

For $M - B = 0.467$:
$$\text{Break-Even Seconds} = \frac{32.4}{0.467} = 69.38 \text{ seconds}$$

> [!NOTE]
> As soon as the user pastes the **Run Time** column from the Kaggle Submissions UI for refs `57165672`, `57253840`, `57260494`, `57267740`, and `57280117`, the exact $E$ values will be populated and ranked.
