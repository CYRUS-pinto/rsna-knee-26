# RSNA Knee Abnormality Detection: Blending & Calibration Mathematics

Comprehensive mathematical manual detailing calibration theory, percentile rank standardization, generalized power-means, Probit normal transformations, and orthopaedic kinetic coupling matrices.

![Specialist Weights](assets/figure2_specialist_weights.png)
![Calibration Curves](assets/figure3_power_probit_calibration.png)

## 1. Metric Mechanics: Why Percentile Rank Standardization is Invariant

The competition evaluation metric is **Macro Mean Area Under the Receiver Operating Characteristic Curve (ROC-AUC)** computed independently across the 12 abnormality columns:

$$
\text{Macro AUC} = \frac{1}{12} \sum_{c=1}^{12} \text{AUC}_c
$$

For any binary label vector $y \in \{0, 1\}^N$ and continuous prediction score vector $\hat{s} \in \mathbb{R}^N$, the ROC-AUC is mathematically equivalent to the Wilcoxon-Mann-Whitney $U$-statistic:

$$
\text{AUC} = \frac{\sum_{i: y_i = 1} \sum_{j: y_j = 0} \mathbf{1}[\hat{s}_i > \hat{s}_j] + \frac{1}{2} \mathbf{1}[\hat{s}_i = \hat{s}_j]}{N_{\text{pos}} \cdot N_{\text{neg}}}
$$

### Mathematical Property of Rank Invariance
Let $g: \mathbb{R} \to \mathbb{R}$ be any strictly monotonically increasing function ($g'(x) > 0$). Then for any pair of samples $(i, j)$:

$$
\hat{s}_i > \hat{s}_j \iff g(\hat{s}_i) > g(\hat{s}_j)
$$

Therefore, $\text{AUC}(y, \hat{s}) = \text{AUC}(y, g(\hat{s}))$.

### The Blending Dilemma with Raw Probabilities
When combining predictions from two distinct model families (e.g., CoAtNet $\hat{s}_1$ and ConvNeXt $\hat{s}_2$), the raw sigmoid probabilities $\sigma(z_1)$ and $\sigma(z_2)$ exhibit fundamentally different calibration curves:
* CoAtNet produces compressed, conservative probabilities clustered in $[0.10, 0.70]$.
* ConvNeXt produces sharp, over-confident probabilities clustered near $0.01$ and $0.99$.

A linear average of raw scores $\hat{s}_{\text{blend}} = w \hat{s}_1 + (1 - w) \hat{s}_2$ allows the steeper, overconfident model to dictate the combined pairwise ordering, effectively nullifying the contribution of the conservative model.

### Empirical Percentile Rank Standardization
To resolve calibration distortion, all raw score vectors are transformed into empirical percentiles over the evaluation cohort:

$$
r_i = \text{RankPct}(x_i) = \frac{\text{rank}(x_i)}{N} \in \left[\frac{1}{N}, 1.0\right]
$$

Percentile ranks enforce identical uniform marginal distributions across all architectures, ensuring that the blend weight $w$ reflects the true desired influence of each backbone.

---

## 2. Generalized Power-Mean Blending (V20 & V60)

While linear rank averaging ($w_1 r_1 + w_2 r_2$) assigns equal marginal importance across the entire range $[0, 1]$, medical diagnostics exhibits asymmetric risk: concordant high-confidence positive detections on focal acute tears (e.g., both models ranking a patient in the top 5% for an ACL tear) represent verified pathognomonic findings that should be non-linearly amplified.

We formulate the **Generalized Power-Mean** with exponent $p$:

$$
M_p(x, y; w) = \left( w \cdot x^p + (1 - w) \cdot y^p \right)^{1/p}
$$

```
+------------------------------------------------------------------------------------+
| MATHEMATICAL PROPERTIES OF THE POWER-MEAN                                          |
+-------------------+--------------------+-------------------------------------------+
| Exponent p        | Formulation        | Diagnostic Behavior                       |
+-------------------+--------------------+-------------------------------------------+
| p = 1.0           | w*x + (1-w)*y      | Pure Linear Rank Mean (Diffusive OA)      |
| p = 1.02 - 1.06   | (w*x^p + ...)^(1/p)| Smooth Focal Calibration (V60 Production) |
| p = 1.40          | (w*x^1.4 + ...)^0.7| Aggressive Focal Amplification (V20 PB)   |
| p -> +infinity    | max(x, y)          | Disjunctive Union (Maximum Sensitivity)   |
+-------------------+--------------------+-------------------------------------------+
```

### Analysis of $p > 1$ on Concordant Detections
Consider two models evaluating a focal lesion (e.g., ACL tear) where Model 1 predicts rank $r_1$ and Model 2 predicts rank $r_2$:
* **Case A (Both High Confidence)**: $r_1 = 0.90, r_2 = 0.90$.  
  $M_{1.4}(0.90, 0.90) = (0.5 \cdot 0.90^{1.4} + 0.5 \cdot 0.90^{1.4})^{1/1.4} = \mathbf{0.90}$.
* **Case B (One High, One Medium)**: $r_1 = 0.90, r_2 = 0.50$.  
  Linear mean ($p=1.0$): $0.5 \cdot 0.90 + 0.5 \cdot 0.50 = 0.7000$.  
  Power mean ($p=1.4$): $(0.5 \cdot 0.90^{1.4} + 0.5 \cdot 0.50^{1.4})^{1/1.4} = \mathbf{0.7412}$.

The power mean smoothly tilts the score toward the higher-confidence detector, preventing a conservative companion model from dragging down a pathognomonic finding, while avoiding heuristic step thresholding.

---

## 3. Probit Normal Transformation & Extreme Tail Calibration

Uniform percentile ranks compress the extreme tails: the difference between the 99th and 99.9th percentile is numerically equivalent to the difference between the 50th and 50.9th percentile ($0.009$). In diagnostic radiology, extreme tail certainty must be preserved.

Using Acklam's rational approximation for the Inverse Normal Cumulative Distribution Function $\Phi^{-1}(p)$, uniform ranks are projected into Gaussian $z$-scores:

$$
z = \Phi^{-1}(p), \quad p \in (0, 1)
$$

### Continuous Probit Ensembling Formulation
We ensemble models directly in Gaussian space and project back via the Gaussian Cumulative Distribution Function $\Phi(z)$:

$$
z_{\text{blend}} = \sum_{m} w_m \Phi^{-1}(r_m), \quad \hat{s}_{\text{probit}} = \Phi(z_{\text{blend}}) = \frac{1}{2} \left[ 1 + \text{erf}\left(\frac{z_{\text{blend}}}{\sqrt{2}}\right) \right]
$$

The continuous hybrid formulation combines:

$$
\hat{s}_{\text{final}} = (1 - \alpha) \cdot \hat{s}_{\text{power\_rank}} + \alpha \cdot \hat{s}_{\text{probit}}
$$

This formulation prevents near-ties from collapsing into discrete step cliffs.

---

## 4. 12x12 Orthopaedic Biomechanical Kinetic Coupling Matrix

![Kinetic Coupling Heatmap](assets/figure4_orthopaedic_coupling.png)

Knee pathologies do not occur in isolation. High-energy pivot-shift trauma forces well-defined kinetic interactions across anatomical structures:

$$
\mathbf{R}_{\text{coupled}} = \mathbf{R} \cdot \mathbf{M}_{\text{coupling}}
$$

| Id | Target Pathology | Source Influences | Clinical Mechanism |
| :---: | :--- | :--- | :--- |
| **0** | **ACL** | Base Prediction | Primary Rupture |
| **1** | **MCL** | Base Prediction | Primary Sprain |
| **2** | **Medial Meniscus** | 98.5% Base + 1.5% Medial OA | Degenerative Maceration |
| **3** | **Lateral Meniscus** | 98.0% Base + 2.0% ACL | Pivot-Shift Impaction |
| **4** | **Medial OA** | 98.5% Base + 1.5% Medial Meniscus | Cartilage Thinning |
| **5** | **Lateral OA** | Base Prediction | Compartment Wear |
| **6** | **PF OA** | 98.0% Base + 1.2% Medial OA + 0.8% Lateral OA | Tricompartmental OA |
| **7** | **Effusion** | 98.5% Base + 1.5% Synovitis | Capsular Hydrops |
| **8** | **Synovitis** | 98.0% Base + 2.0% Effusion | Reactive Synovium |
| **9** | **Baker's Cyst** | 97.5% Base + 1.5% Effusion + 1.0% Medial Meniscus | Valve Extravasation |
| **10** | **Bone Contusion** | 97.0% Base + 3.0% ACL | Trabecular Bruise |
| **11** | **Bone Fracture** | 97.5% Base + 1.5% Bone Contusion + 1.0% Effusion | Cortical Disruption |

### Invariant Guarantee of Continuous Coupling
Unlike early prototype implementations that called `rank_matrix()` after coupling (which discretized continuous scores), our production formulation updates scores linearly:

$$
R_c' = \left(1 - \sum \lambda_k\right) R_c + \sum \lambda_k R_k
$$

Because $\sum \lambda_k < 0.03$, the original model rankings are preserved while physiologically concordant ties are cleanly resolved in the correct kinetic direction.
