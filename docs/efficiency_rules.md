# RSNA Knee Abnormality Detection: Efficiency Prize Specification

**Status:** UNVERIFIED / PENDING HOST RULES TEXT  
**Author:** Senior ML Engineering  
**Evidence Standard:** MEASURED / ESTIMATED / CLAIMED  

---

## 1. Metric Formula

| Element | Status | Formula / Definition | Source / Evidence |
| :--- | :--- | :--- | :--- |
| **Claimed Efficiency Formula** | CLAIMED (Unverified) | $E = \frac{\text{AUC}}{B - M} + \frac{T}{32,400}$ | User prompt / `validation/official_metric.py` line 112 |
| **Benchmark AUC ($B$)** | UNKNOWN | Numerical value not officially extracted | Rules page pending user paste |
| **Max AUC ($M$)** | UNKNOWN | Numerical value not officially extracted | Rules page pending user paste |
| **Denominator ($B - M$)** | ESTIMATED | Estimated $\approx 0.467$ ($M - B$) in prompt | Unverified prompt text |
| **Runtime Unit ($T$)** | CLAIMED | Wall-clock seconds of hidden test evaluation run | Standard Kaggle efficiency competition protocol |
| **Direction** | CLAIMED | Minimization ($E$ lower is better) | `validation/official_metric.py` line 114 |

---

## 2. Sensitivity Analysis (Under Claimed Formula)

Under the claimed formula:
$$\Delta E = \frac{\Delta \text{AUC}}{B - M} + \frac{\Delta T}{32,400}$$

Assuming $M - B \approx 0.467$:
$$\frac{\Delta \text{AUC}}{0.467} \approx \frac{\Delta T}{32,400} \implies \Delta T \approx 32,400 \times \frac{\Delta \text{AUC}}{0.467} \approx 69,379 \times \Delta \text{AUC}$$

* Break-even seconds for $+0.001$ AUC:
  $$\Delta T = 69,379 \times 0.001 \approx 69.4 \text{ seconds}$$
* **Implication (ESTIMATED):** Gaining $+0.001$ AUC is mathematically neutralized if it adds $> 69.4$ seconds to test runtime. Conversely, saving 1 hour ($3,600$s) is mathematically equivalent to gaining:
  $$\Delta \text{AUC} = \frac{3,600}{69,379} \approx +0.0519 \text{ AUC}$$

---

## 3. Eligibility & Operational Constraints

| Constraint | Status | Details |
| :--- | :--- | :--- |
| **Prize Pool** | CLAIMED | $18,000 ($7,000 / $6,000 / $5,000) across top 3 eligible efficiency submissions |
| **Leaderboard Qualification** | UNKNOWN | Whether a team must finish in the Top X% of standard leaderboard to be eligible |
| **Selection Requirement** | CLAIMED | Submission must be one of the two final selected submissions |
| **Maximum Hard Limit** | MEASURED | 9.0 hours (32,400 seconds) hard timeout on Kaggle GPU notebook execution |
| **External Data Exclusions** | UNKNOWN | Whether OAI pre-training is permissible under efficiency track |

---

## 4. Pending Required Inputs

> [!IMPORTANT]
> To upgrade this document from `CLAIMED` to `MEASURED`:
> 1. User must paste the exact raw text or equation image from the Kaggle Competition **Evaluation** tab.
> 2. User must paste the `Run Time` column from the Kaggle **Submissions** page for V6, V17, V18, V20, V22.
