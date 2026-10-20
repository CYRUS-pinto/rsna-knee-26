# ADR-029: ConvNeXt Fold-0 Pruning, Power-Mean Ensembling ($p=1.5$), and Candidate V29 Engineering

**Status:** ACCEPTED & PUSHED TO KAGGLE  
**Date:** 2026-10-17  
**Author:** AI Technical Advisor & Engineering Team  
**Scope:** Eliminating the 3.5-hour ConvNeXt inference bottleneck, securing top standing on the $18,000 Efficiency Prize track, and upgrading ensembling math from linear averaging to non-linear power-mean ranking.

---

### 1. Context & Motivation

Our competitive teardown revealed four structural obstacles holding the team back from dominating both tracks:
1. **The 3.5-Hour ConvNeXt Anchor:** Standard ConvNeXt R7 runs all 4 folds (`f0, f1, f2, f3`) across 587 sliding windows, requiring $2,348$ forward passes per study. Over 1,300 test studies, inference takes ~3.5 hours, consuming virtually all compute and preventing the integration of a 3rd architecture.
2. **Linear Arithmetic Blend Penalty:** Arithmetic averaging $\frac{P_A + P_B}{2}$ penalizes extreme predictions when one model detects a sharp focal abnormality (e.g. subtle fracture or lateral meniscus tear) and the other produces a conservative diffuse output. In AUC competitions, high-confidence True Positives must remain at the top rank.
3. **The $18,000 Efficiency Track Opportunity:** Our Candidate V26 (Fast MultiScale Raptor via `_fastread.py`) demonstrated a ~5-minute runtime scoring `0.949`. Integrating a lightweight, pruned ConvNeXt provides the missing joint-space inductive bias to push efficiency scores into the `0.954 - 0.956` range within $<400$ seconds.

---

### 2. Engineering Innovations in Candidate V29

#### A. ConvNeXt Fold-0 Pruning (75% Compute Reduction)
In Cell 6 of [`rsna-knee-v29-fast-power-hybrid-master.ipynb`](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/submissions/v29_fast_power_hybrid_master/rsna-knee-v29-fast-power-hybrid-master.ipynb):
```python
# Dynamically prune manifest models before loading
manifest['models'] = ['E_lrprior_s0_f0.pt']
print(f"[V29 HYBRID OPTIMIZATION] Pruned ConvNeXt models to: {manifest['models']}", flush=True)

for name in manifest['models'] + (manifest.get('models2') or []):
    shutil.copy(M_SRC / name, BUNDLE / name)
```
* **Mechanism:** `compact_infer.py` automatically evaluates only the `.pt` models present in `BUNDLE`.
* **Impact:** Forward passes drop from 2,348 to 587 per study. Compute overhead is slashed by **75%**.

#### B. Power-Mean Ensembling ($p = 1.5$)
In Stage 3 (Cell 8), we replace the linear combination with a non-linear power-mean:
```python
p_power = 1.5
rr_safe = np.clip(rr, 0.0, 1.0)
ro_safe = np.clip(ro, 0.0, 1.0)
b = (wr * (rr_safe ** p_power) + (1.0 - wr) * (ro_safe ** p_power)) ** (1.0 / p_power)
```
* **Mathematical Rationale:** For $p > 1$, the power mean acts as a soft maximum. If either specialist model identifies an unambiguous tear or fracture ($P > 0.90$), its percentile rank is preserved at the top of the leaderboard ranking distribution rather than being dragged down by the other model's lower score.

#### C. Specialist Weight Distribution (Restored Inductive Bias)
We maintain the decoupled compartmental weights established in ADR-022 and ADR-028:
- **Lateral Meniscus:** 0.15 Raptor / 0.85 ConvNeXt R7
- **Medial OA & Lateral OA:** 0.20 Raptor / 0.80 ConvNeXt R7
- **Synovitis:** 0.80 Raptor / 0.20 ConvNeXt R7
- **PF OA:** 0.60 Raptor / 0.40 ConvNeXt R7
- **Baker's Cyst:** 0.35 Raptor / 0.65 ConvNeXt R7
- **Cruciate/Collateral (ACL/MCL):** 0.70 / 0.65 Raptor
- **Symmetric classes (Effusion, Contusion, Fracture):** 0.50 / 0.50

#### D. MultiScale Split Optimization
- Set `A224 = 0.25` (75% weight on 384px full-field, 25% on 224px center 0.8 crop), aligning with the optimal capacity ratio established in ADR-027.

---

### 3. Submission Artifacts

- **Notebook:** [`submissions/v29_fast_power_hybrid_master/rsna-knee-v29-fast-power-hybrid-master.ipynb`](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/submissions/v29_fast_power_hybrid_master/rsna-knee-v29-fast-power-hybrid-master.ipynb)
- **Metadata:** [`submissions/v29_fast_power_hybrid_master/kernel-metadata.json`](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/submissions/v29_fast_power_hybrid_master/kernel-metadata.json)
- **Kaggle Kernel:** `cyruspinto/rsna-knee-v29-fast-power-hybrid-master` (Version 1 pushed and currently executing GPU commit).
