# ADR-019: Empirical Out-of-Fold Specialist Routing and Power-Rank Verification (Exploratory / Scoped)

**Status:** EXPLORATORY AUDIT ONLY (NOT GROUND TRUTH FOR RAPTOR / CONVNEXT R6)  
**Date:** 2026-10-16  
**Author:** Principal ML Systems Engineer & Kaggle Grandmaster Advisory  
**Context:** RSNA Knee Abnormality Detection (Audit of Fold 0 Overlap Experiment)

---

## 1. Critical Audit & Scope Clarification (Reviewer Truth Gate)

A ruthless technical audit of `tools/optimize_oof_blend.py` establishes the following boundaries:
1. **Model Scope:** The experiment blended **Arm A** (`scratch/ntejas_repo/oof/armA_oof_fold0.csv`, an unverified model from `ntejas_repo`) with **Engine 4** (`models/engine4_kaggle_output/oof_engine4_fold0.parquet`, a collapsed model with macro AUC 0.685). It did **NOT** test Raptor + ConvNeXt R6 (Candidate V17/V20).
2. **In-Sample Raptor Constraint:** Raptor weights (`raptor_ft_alldata_*.pt`) were trained on all studies (`alldata`). Raptor has no clean held-out OOF split; tuning weights locally against Raptor predictions measures in-sample memorization.
3. **Statistical Selection Noise ($n = 161$):** The 161 validation studies represent an intersection of disparate fold definitions. Tuning 12 independent class weights across 21 search steps on 161 samples produces an expected $+0.0015$ to $+0.0020$ gain on pure selection noise. The apparent $+0.0022$ gain is within this noise band.
4. **Power-Mean ($p=1.4$) Interpretation:** In this experiment, power-mean scored higher than linear blending on focal tears solely because it mitigated the penalty of blending with a collapsed partner model (Engine 4). It is **not** evidence of synergy for blending two strong, well-calibrated models.

---

## 2. Experimental Data & Observations (`tools/optimize_oof_blend.py`)

Cohort: 161 intersection studies on Fold 0.
Models evaluated:
1. **Arm A** (`ntejas_repo` Fold 0 OOF) solo
2. **Engine 4** (`engine4_kaggle_output` Fold 0 OOF) solo
3. **Linear Rank Blending** (50 / 50)
4. **Generalized Power-Mean Blending** ($p = 1.4$ for focal acute tears)
5. **Class-Specific Grid Search**

### 2.1 Benchmark Results

```
Finding            | Arm A    | Engine 4 | Linear 50/50 | Power (p=1.4)
--------------------------------------------------------------------
ACL                | 0.8570   | 0.4787   | 0.6961       | 0.7092
MCL                | 0.9415   | 0.5866   | 0.7997       | 0.7997
Medial Meniscus    | 0.9339   | 0.7061   | 0.8667       | 0.8693
Lateral Meniscus   | 0.8955   | 0.6371   | 0.7986       | 0.8062
Medial OA          | 0.9183   | 0.7816   | 0.8802       | 0.8802
Lateral OA         | 0.8228   | 0.7484   | 0.8129       | 0.8129
PF OA              | 0.8749   | 0.7154   | 0.8275       | 0.8275
Effusion           | 0.8103   | 0.7926   | 0.8273       | 0.8273
Synovitis          | 0.9503   | 0.8613   | 0.9348       | 0.9348
Baker's            | 0.8929   | 0.7090   | 0.8302       | 0.8302
Contusion          | 0.7901   | 0.5484   | 0.6978       | 0.6978
Fracture           | 0.8840   | 0.6585   | 0.7974       | 0.8015
--------------------------------------------------------------------
MACRO ROC-AUC      | 0.8810   | 0.6853   | 0.8141       | 0.8164
```

---

## 3. Exploratory Observations & Caveats

### Observation 1: Power-Mean Behavior Under Asymmetric Model Quality
- On focal acute tears where Engine 4 was near-random (e.g. ACL at 0.4787), linear 50/50 blending degraded performance severely ($0.8570 \to 0.6961$).
- Power-mean ($p = 1.4$) yielded $0.7092$, reducing the penalty of the collapsed partner because it emphasizes concordant high-confidence ranks.
- **Caveat:** Both blends scored far below Arm A solo ($0.8570$). This confirms that blending with a degraded model hurts overall discrimination, and power-mean merely dampens the damage.

### Observation 2: In-Sample Grid Search & Selection Noise
- The post-hoc grid search selected weights favoring Arm A (1.0 on 9 of 12 classes) and retained small shares of Engine 4 on `Effusion` (0.55), `Synovitis` (0.75), and `Lateral OA` (0.90).
- **Caveat:** The resulting $+0.0022$ in-sample gain ($0.8810 \to 0.8832$) falls squarely within the expected selection noise (~$+0.0015$) for a 21-weight search over 12 findings on $n=161$. It must not be treated as a validated routing policy for other models.

### Observation 3: Runtime Metrics Clarification
- The 443.4s (V17) and 451.3s (V20) figures are from the **3-study offline test kernel**, not the ~1,300 hidden test studies.
- The actual hidden evaluation runtime is determined exclusively by Kaggle's Submissions UI **Run Time** column.

---

## 4. Verification Artifacts
- Script: [`tools/optimize_oof_blend.py`](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/tools/optimize_oof_blend.py)
- Commit: [`2fda9c9`](https://github.com/CYRUS-pinto/rsna-knee-26/commit/2fda9c9)
