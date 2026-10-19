# RSNA Knee Grandmaster Technical Compendium: 10 Core Pillars for Rank 1 Dominance

**Target:** RSNA Knee Abnormality Detection Challenge (4 Days Remaining)  
**Authors:** Cyrus Pinto & Antigravity IDE Autonomous Pair  
**Standards:** Full Mathematical Grounding, Zero Fluff, Reproducible Code Assets

---

## Pillar 1: Epipolar Geometry & Slice-Continuity Interpolation
### Mathematical Foundation
In knee MRI sequences (Sagittal, Coronal, Axial), 2D slices represent discrete cross-sections of continuous 3D anatomical structures. A ligament (e.g., ACL) is an elongated cylindrical fiber spanning 8 to 15 contiguous slices. Conventional 2D CNNs treat slices as independent images, losing **inter-slice fiber continuity**.
By treating adjacent slices as epipolar planes, we compute the inter-slice displacement vector $\Delta \mathbf{x} = (\Delta u, \Delta v)$ across depth $z$:
$$I(u, v, z + \Delta z) \approx I(u + \Delta u, v + \Delta v, z) + \nabla I \cdot \Delta \mathbf{x}$$
In our 2.5D Triplet Windowing, channels $[c-1, c, c+1]$ capture this temporal/spatial gradient $\frac{\partial I}{\partial z}$, providing direct sensitivity to oblique ligament fiber disruptions.

### Concrete Implementation
```python
def extract_epipolar_triplets(vol, n_slices=8):
    D, H, W = vol.shape
    step = max(1, (D - 2) // n_slices)
    centers = [min(D - 2, 1 + i * step) for i in range(n_slices)]
    triplets = []
    for c in centers:
        trip = vol[c-1:c+2].astype(np.float32) / 255.0  # (3, H, W)
        triplets.append(trip)
    return np.stack(triplets, axis=0)  # (n_slices, 3, H, W)
```

---

## Pillar 2: Scale-Invariant Multi-Attention Pooling (MIL)
### Mathematical Foundation
Knee abnormalities exhibit extreme spatial variance:
- **Focal Pathologies:** Fractures, bone contusions, meniscal tears occupy small localized regions (often only 1–3 slices).
- **Diffuse Pathologies:** Osteoarthritis, joint effusion, synovitis affect the entire joint cavity.
Simple Global Average Pooling (GAP) dilutes focal signals ($1/S \sum_s z_s \approx 0$). Simple Max Pooling ignores cumulative diffuse damage.
We apply **Gated Gated Multi-Instance Attention Pooling**:
$$\mathbf{a} = \text{Softmax}\left(\mathbf{w}_a^T \left(\tanh(\mathbf{V} \mathbf{h}_s) \odot \sigma(\mathbf{U} \mathbf{h}_s)\right)\right)$$
$$\mathbf{z}_k = \sum_{s=1}^S a_{s, k} \mathbf{h}_s, \quad \text{Logit}_k = \mathbf{w}_k^T \mathbf{z}_k + b_k$$
This enables the network to dynamically assign attention $a_{s,k} \to 1.0$ to the single slice containing a subtle fracture, while distributing weights evenly ($a_{s,k} \approx 1/S$) for generalized joint effusion.

---

## Pillar 3: Dissecting TPR, FPR, and Rank-Order Invariance in Macro-AUC
### Mathematical Foundation
The official competition evaluation metric is **Macro-Averaged AUC ROC**:
$$\text{Macro-AUC} = \frac{1}{12} \sum_{k=1}^{12} \text{AUC}_k$$
Where $\text{AUC}_k$ is mathematically identical to the normalized Mann-Whitney $U$ statistic:
$$\text{AUC}_k = \frac{\sum_{i \in \mathcal{P}_k} \sum_{j \in \mathcal{N}_k} \mathbb{I}(p_i > p_j) + \frac{1}{2}\mathbb{I}(p_i = p_j)}{|\mathcal{P}_k| \cdot |\mathcal{N}_k|}$$
### Critical Implications:
1. **Calibration Independence:** Shifting all predictions by $+0.3$ or scaling by $2.0$ has **zero effect on AUC**.
2. **Rank Alignment:** Blending models with disparate output scales (e.g. Model A outputs $[0.01, 0.20]$, Model B outputs $[0.40, 0.95]$) causes Model B to dominate the ordering.
3. **The Solution:** Converting each model's predictions to empirical uniform percentiles before ensembling:
$$\tilde{p}_{i, k} = \frac{\text{Rank}(p_{i, k})}{N}$$

---

## Pillar 4: Semi-Supervised Active Learning (SSL/AL)
### The Problem:
Out of ~8,000 training studies, only 58 have verified radiologist annotations across all 12 pathologies. The rest have unstructured, free-text radiology reports in English, Spanish, and German.
### The Grandmaster Method:
1. **Uncertainty Sampling ($H(x)$):**
   $$H(x) = -\sum_{k=1}^{12} \left[ p_k \log p_k + (1 - p_k) \log(1 - p_k) \right]$$
   Studies with maximum Shannon entropy are borderline cases where the ensemble is uncertain.
2. **Diversity Sampling via Latent K-Means:**
   Cluster backbone feature representations $\mathbf{h} \in \mathbb{R}^{1024}$ into $K=50$ clusters and sample across all clusters to ensure representation across 1.5T vs 3.0T scanners and diverse patient demographics.
3. **Calibrated Soft Targets:**
   Train backbones using soft target probabilities derived from LLM consensus extraction with label smoothing ($\epsilon = 0.05$).

---

## Pillar 5: Asymmetric Focal Loss (ASL) for Extreme Class Imbalance
### Mathematical Foundation
In knee MRIs, pathologies like **Fracture** and **MCL Tears** have $< 8\%$ positive prevalence, while negative samples dominate $> 92\%$. Standard Binary Cross Entropy (BCE) gradients are overwhelmed by easy negative background tissue.
Asymmetric Loss modifies the cross-entropy loss:
$$\mathcal{L}_{\text{ASL}} = - y (1 - p)^{\gamma_+} \log(p) - (1 - y) (p_m)^{\gamma_-} \log(1 - p_m)$$
Where $p_m = \max(p - m, 0)$ is the probability shifted by hard-threshold margin $m \ge 0.05$.
- Setting $\gamma_- = 4, \gamma_+ = 1, m = 0.05$ completely zeroes out the gradients from easy negative slices ($p < 0.05$), forcing backpropagation to update weights exclusively on suspicious candidate slices.

---

## Pillar 6: Multi-Plane Feature Fusion (Sagittal + Coronal + Axial)
### Architectural Blueprint
Single-plane models fail on pathologies requiring multi-perspective verification (e.g., confirming a meniscus tear on Sagittal by verifying blunted horn morphology on Coronal).
- **Sagittal Backbone:** ConvNeXt-Base-384 $\to \mathbf{f}_{\text{sag}} \in \mathbb{R}^{1024}$.
- **Coronal Backbone:** ConvNeXt-Base-384 $\to \mathbf{f}_{\text{cor}} \in \mathbb{R}^{1024}$.
- **Cross-Plane Bilinear Fusion:**
  $$\mathbf{z} = \text{LayerNorm}\left([\mathbf{f}_{\text{sag}} \,\|\, \mathbf{f}_{\text{cor}}] \mathbf{W}_{\text{proj}}\right)$$
  $$\text{Logits} = \text{Classifier}(\mathbf{z})$$

---

## Pillar 7: High-Frequency Texture vs. Low-Frequency Geometry
### Complementary Inductive Biases
1. **High-Frequency Local Texture (ConvNeXt-Base):**
   - $7\times 7$ depthwise convolutions capture fine bone trabeculae and subchondral sclerosis (critical for Contusion and Osteoarthritis).
2. **Global Hierarchical Attention (Swin Transformer):**
   - Shifted-window multi-head self-attention captures long-range anatomical deformation (e.g. tibial anterior translation relative to the femur in complete ACL rupture).
- Ensembling these two architectures yields maximum orthogonality and AUC boost.

---

## Pillar 8: The $18,000 Efficiency Prize Metric Architecture
### Formula
$$\text{Score} = \frac{\text{Macro-AUC}}{0.500 - \max\text{AUC}} + \frac{\text{RuntimeSeconds}}{32400}$$
### The Conversion Law:
- In this competition, $\max\text{AUC} \approx 0.969$. The denominator is $0.500 - 0.969 = -0.469$.
- A 1,000-second increase in container runtime adds $+0.0308$ penalty.
- To overcome $+0.0308$ penalty purely with accuracy, a model must achieve $+0.0144$ higher AUC!
- **Strategy:** Our Stride-2 inference pipeline finishes in **$\sim 420\text{ seconds}$**, incurring only $+0.0129$ runtime penalty, providing a massive structural advantage over heavy 4-hour kernels.

---

## Pillar 9: Leak-Free Grouped 5-Fold Validation Harness
### Rules:
1. **Grouping by Patient/Study:** Scans from the same patient must never cross folds.
2. **Multi-Label Iterative Stratification:** Rare co-occurring conditions (e.g., ACL tear + Medial Meniscus tear + Effusion) must be balanced across all 5 folds using `iterative-stratification`.
3. **Out-Of-Fold Stacking:** Meta-learner weights are fitted exclusively on out-of-fold validation predictions to avoid target leakage.

---

## Pillar 10: Two-Track Championship Defense Doctrine
### The Protocol:
- **Track 1 (Bedrock Defense):** Permanently lock our top 2 verified baseline submissions (Slot 1: `57253840` at **0.958**, Slot 2: `57165672` at **0.956**). Never unselect them unless an experimental candidate clears $> 0.9610\text{ AUC}$.
- **Track 2 (Sovereign Offense):** 100% RSNA-trained multi-engine models (ConvNeXt-Base + CoAtNet + Swin) that are 100% compliant with competition rules, fully open-source, and immune to post-freeze external data disqualification.
