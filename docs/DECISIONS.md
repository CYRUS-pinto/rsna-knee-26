# RSNA Knee Abnormality Detection: Architectural Decision Records (ADR-001 to ADR-049)

Official log of immutable architectural decisions, experimental validations, and rejected hypotheses governing the engineering lifecycle.

| ADR ID | Title / Decision Scope | Status | Metric Impact |
| :---: | :--- | :---: | :--- |
| **ADR-001** | **Empirical Percentile Rank Standardization** | ACCEPTED | Invariant Gain |
| **ADR-002** | **Excising Heuristic IF/ELSE Conditional Rules** | ACCEPTED | +0.0010 AUC |
| **ADR-015** | **3D Physical Normal Slice Sorting (n = r x c)** | ACCEPTED | +0.0280 AUC |
| **ADR-016** | **140mm Metric Physical Bounding Box Cropping** | ACCEPTED | Scale Invariance |
| **ADR-028** | **Generalized Power-Mean (p=1.4) on Focal Tears**| ACCEPTED | 0.9420 Private LB |
| **ADR-037** | **Fail-Soft Zero-Fatal-Assert Policy** | ACCEPTED | Zero Crash Guarantee |
| **ADR-048** | **FP16 Weight-Space Model Soups (<560 MB)** | ACCEPTED | Resolves 20GB Quota |
| **ADR-049** | **Continuous Probit Ensembling (Phi^-1)** | ACCEPTED | +0.0015 AUC Lift |

---

## Key Decision Details

### ADR-001: Percentile Rank Standardization
* **Context**: Blending raw sigmoids allows over-confident models to overpower conservative models.
* **Decision**: All prediction vectors must be transformed to empirical percentiles: $\text{RankPct}(x) \in [1/N, 1.0]$.
* **Status**: ACCEPTED across all production pipelines.

### ADR-002: Rejection of Hand-Coded Heuristic Threshold Floors
* **Context**: Hand-coded if/else rules (V9-V11) degraded LB by -0.0010 due to non-smooth step cliffs.
* **Decision**: All physiological priors must operate as continuous linear or power-mean updates.
* **Status**: ACCEPTED (codified in V12 clean gate).

### ADR-015: 3D Normal Physical Slice Sorting
* **Context**: Lexical sorting scrambled through-plane MRI slice continuity.
* **Decision**: Calculate slice normal $n = \text{IOP}_{0:2} \times \text{IOP}_{3:5}$ and sort along depth $d = \text{IPP} \cdot n$.
* **Status**: ACCEPTED (+0.0280 AUC leap in V6).

### ADR-028: Generalized Power-Mean Blending on Focal Acute Lesions
* **Context**: Focal tears require non-linear amplification of concordant high-confidence signals.
* **Decision**: Formulate $M_p(x, y; w) = (w \cdot x^p + (1 - w) \cdot y^p)^{1/p}$ with $p = 1.02\text{ to }1.40$ on focal acute lesions.
* **Status**: ACCEPTED (delivered 0.9420 Private LB in V20).

### ADR-048: Weight-Space Model Soup & FP16 Quantization
* **Context**: Attaching 8 separate FP32 datasets breached Kaggle's 20 GB container mount limit.
* **Decision**: Average multi-fold checkpoints in weight-space and cast to FP16 (<560 MB).
* **Status**: ACCEPTED (resolved container mount failures in V50+).

### ADR-049: Continuous Probit Quantile Ensembling
* **Context**: Calling `rank_matrix()` multiple times sequentially created discrete step cliffs ($0.3333, 0.6667, 1.0000$).
* **Decision**: Ensemble in continuous Gaussian space via Acklam's inverse normal CDF $\Phi^{-1}(r)$, followed by CDF projection $\Phi(z)$.
* **Status**: ACCEPTED (codified in Candidate V60).
