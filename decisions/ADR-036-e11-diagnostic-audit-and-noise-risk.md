# ADR-036: Empirical 100-Study Diagnostic Results — E11 Noise Exposure & Twin Ablation Imperative

## Status
**APPROVED & ACTIVE** — October 17, 2026

---

## 1. Context & Diagnostic Execution
To empirically verify whether the RadImageNet ResNet-50 E11 arm reproduces its offline training signal before burning a competition submission, we executed a standalone 100-study diagnostic kernel on Kaggle CPU (`cyruspinto/rsna-knee-e11-100-study-diagnostic`).

### The Design:
* Evaluated **100 stratified training studies** across their exact held-out fold heads (`head_models[fold_idx]`).
* Computed Spearman rank correlation ($\rho$) against official out-of-fold ground truth (`v52_e11_oof.csv`).
* Evaluated an adversarial **Negative Control** (swapped Fat-Suppressed $\leftrightarrow$ Non-Fat-Suppressed routing).

---

## 2. Empirical Findings

The diagnostic completed in **668.0s** (~6.68s / study on CPU, scaling to ~1.5s on GPU):

```
--- SPEARMAN CORRELATION VS OOF TRUTH ---
Finding                 Standard E11   Negative Control (Swapped)
  ACL               :         0.5733                      0.4376
  Baker's           :         0.4437                      0.3348
  Contusion         :         0.3159                      0.2111
  Effusion          :         0.3738                      0.4294
  Fracture          :         0.1977                      0.0923
  Lateral Meniscus  :         0.3829                      0.2936
  Lateral OA        :         0.7067                      0.6132
  MCL               :         0.5254                      0.4309
  Medial Meniscus   :         0.3946                      0.3693
  Medial OA         :         0.5774                      0.5634
  PF OA             :        -0.0493                     -0.0230
  Synovitis         :         0.2947                      0.2943

STANDARD E11 MEAN SPEARMAN: 0.3947 (Min: -0.0493, Labels < 0.70: 11)
NEGATIVE CONTROL MEAN SPEARMAN: 0.3372
```

### Forensic Analysis & The Smoking Gun:
1. **`v52_e11_oof.csv` Contains E11 Model Predictions (Not Labels):**
   * Inspection of `v52_e11_oof.csv` confirms it stores continuous probabilities (4,407 unique float values per target, min $\approx 0.009$, max $\approx 0.998$).
   * Therefore, the diagnostic was testing whether the inference code reproduces E11's own training-time predictions on the identical 100 studies and fold heads.
   * **The $\rho = 0.3947$ result is proof of an inference pipeline mismatch/bug** (e.g. DICOM windowing, slice selection, or normalization discrepancy) between how E11 was trained and how it was inferred in the notebook.
   * This is far worse than a weak signal: it means the inference arm is broken and cannot reproduce its training outputs.
2. **Adversarial Negative Control:**
   * The scrambled negative control (swapped FS $\leftrightarrow$ Non-FS routing) scored $\rho = 0.3372$ because anatomical and fluid series share correlated features. Standard E11 scored only $+0.0575$ above it.
3. **Catastrophic Inversions:**
   * **PF OA:** $\rho = -0.0493$ (negatively correlated!).
   * **Synovitis:** $\rho = 0.2947$ (identical to swapped routing $0.2943$).
   * **11 out of 12 labels fall below 0.70 correlation.**

---

## 3. Strategic Decision: Reject E11 and Protect Finals

1. **E11 is Broken in Inference:** We will NOT inject E11 into our final submissions. Every weak or misaligned third arm in this competition (Engine 4, Engine 5, Engine 6) has degraded Public LB.
2. **Stride-2 Twin Role:** Stride-2 only trades accuracy for speed. With efficiency prize dropped, Stride-2 has no purpose for the main track.
3. **Final Selection Rules (Strict Threshold):**
   * **Slot 1:** **Candidate V17 (`57253840`, 0.958)**.
   * **Slot 2:** **Candidate V6 (`57165672`, 0.956)** or **Candidate V20 (`57267740`, 0.958)**. V6 provides genuine model diversity through the ConvNeXt R7 checkpoint.
   * **Displacement Gate:** Do NOT displace V17 or V6/V20 unless an experimental submission scores **$\ge 0.961$**. No chasing 0.001 noise.
