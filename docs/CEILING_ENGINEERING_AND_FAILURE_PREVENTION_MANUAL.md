# Ceiling Engineering & Failure Prevention Manual: Road to Top 1 (0.969)

**Target:** RSNA Knee Abnormality Detection Challenge (2026)  
**Standard:** World #1 Grandmaster Architecture & Absolute Domain Ceiling  
**Author:** Antigravity AI Chief Orthopaedic Imaging Architect  
**Timestamp:** 2026-10-20 15:55 UTC (Deadline: 2026-10-22 23:59 UTC — ~56h remaining)  

---

## 1. What Went Wrong & Why: Forensic Autopsy of Historical Failures

To achieve the absolute top tier, we must confront reality with unsparing intellectual honesty. We analyze every single historical failure in this campaign:

```
Historical Failure Forensic Summary:
┌─────────────────┬───────────┬────────────┬──────────────────────────────────────────────────────────────┐
│ Candidate / Run │ Result    │ Score Drop │ Root Cause Mechanism                                         │
├─────────────────┼───────────┼────────────┼──────────────────────────────────────────────────────────────┤
│ V47             │ 0.953     │ -0.005     │ Spatial Nyquist Blur: Double-blended 224px (effective 57.8%) │
│ V42             │ 0.943     │ -0.015     │ Power Sharpening (p=1.4) distorted uniform rank distribution │
│ V34–V36         │ ERROR     │ Outage     │ Unconditional assert on hidden test length (crashed worker)  │
│ V12             │ 0.953     │ -0.005     │ Hand-tuned heuristic priors without full OOF validation     │
│ V18–V33         │ 0.956–958 │ Plateau    │ Pseudo-diversity: Re-weighting identical models (rho > 0.95) │
│ Single-Fold     │ High Var  │ Potential  │ E12 & E13 trained on Fold 0 only (uncalibrated OOF spread)    │
└─────────────────┴───────────┴────────────┴──────────────────────────────────────────────────────────────┘
```

### 1.1 Candidate V47 Regression (0.958 → 0.953)
- **What Happened:** Candidate V47 dropped from the 0.958 plateau down to 0.953.
- **Why It Happened:** Recursive double-blending. The script loaded `raptor_sub.csv` (which was already an internal ensemble of 384px and 224px slices) and blended it with `raptor224_sub.csv` at weight 0.30. This caused the net mathematical weight of low-resolution $224 \times 224$ images to reach **57.8%** of the entire prediction.
- **Biomechanical Impact:** Lesions in the knee (e.g., subtle radial meniscus tears, posterior horn root tears, undisplaced tibial plateau fractures) measure between $0.8\text{ mm}$ and $2.0\text{ mm}$. Downsampling native $512 \times 512$ MRI DICOM acquisitions to $224 \times 224$ results in pixel spacing of ~1.8 mm, causing spatial aliasing and Nyquist blur that obliterates the physical signal of these tears.
- **How We Prevented It:** Formulated **Rule 11** and automated it in `tools/validate_kernel_guardrail.py`. Any submission where native $\ge 384\text{px}$ backbones represent $< 70\%$ of the blend or where redundant constituent files are re-blended is blocked from deployment.

### 1.2 Candidate V42 Collapse (0.958 → 0.943)
- **What Happened:** Candidate V42 suffered a massive -0.015 drop down to 0.943.
- **Why It Happened:** Application of non-linear power rank sharpening $S_i = \left(\sum_k w_k r_{i,k}^p\right)^{1/p}$ with exponent $p = 1.4$.
- **Mathematical Impact:** Macro ROC AUC is invariant only under strictly monotonic increasing transformations applied uniformly across all classes. Applying power sharpening non-linearly distorts the probability density, compressing low-probability true negatives into near-zero clusters. This induces widespread rank ties and inversions in the negative distribution, severely penalizing the Wilcoxon-Mann-Whitney U-statistic:
  $$\text{AUC} = \frac{1}{N^+ N^-} \sum_{i \in \mathcal{P}} \sum_{j \in \mathcal{N}} \mathbb{I}(S_i > S_j) + \frac{1}{2} \mathbb{I}(S_i = S_j)$$
- **How We Prevented It:** Enforced **Rule 2**: Linear rank-percentile blending ONLY ($\sum w_k r_k$ with $\sum w_k = 1.0$). Zero power transforms permitted.

### 1.3 Candidates V34–V36 Fatal Exception Outages (Kaggle Error)
- **What Happened:** 3 consecutive daily submissions failed on Kaggle with "Notebook Threw Exception" before producing a score.
- **Why It Happened:** Hard assertions on test data dimensions (`assert len(df) == 120` and `assert series_count == 3`). While the interactive sample test set contains 3 studies with 120 rows, the hidden private test set contains thousands of studies, some of which feature non-standard DICOM tags or missing series.
- **How We Prevented It:** Codified **Rule 4**: Zero fatal assertions on test data. All DICOM extraction and series parsing are wrapped in guarded try/except blocks with column-median prior fallback.

### 1.4 Candidate V12 Heuristic Prior Degradation (0.958 → 0.953)
- **What Happened:** Hand-tuned clinical conditional boost rules dropped score to 0.953.
- **Why It Happened:** Radiologist intuition suggested that if an ACL tear is detected, bone contusion probability should be multiplied by $1.3\times$. However, uncalibrated multiplicative boosting uniformly raises false positive rates across borderline cases, degrading AUC.
- **How We Prevented It:** Codified **Rule 3**: No hand-crafted clinical boost rules without empirical proof on full 4,407-study 5-fold OOF ($\Delta \text{AUC} \ge +0.002$).

### 1.5 The 0.956–0.958 Stagnation Plateau (V18–V33)
- **What Happened:** 15 submissions spent days hovering between 0.956 and 0.958.
- **Why It Happened:** Pseudo-diversity. Re-blending the same two public models (CoAtNet and ConvNeXt-Small) that had a Spearman rank correlation $\rho > 0.95$. Blending two highly correlated models yields zero variance reduction.
- **How We Prevented It:** Trained genuinely orthogonal architectures: **Engine 12 (ConvNeXt-Large-384, Gold AUC 0.8808)** and **Engine 13 (Swin-Base-384, Gold AUC 0.8577)** to introduce true model diversity.

---

## 2. Limitations We Must Overcome to Reach Top 1 (#1 / 0.969)

1. **The Public Notebook Plateau (0.958):**
   - 1,600 teams tie at $\ge 0.958$. Anyone who relies on public weights will remain stuck here. Breaking out requires proprietary high-capacity models (ConvNeXt-Large + Swin-Base + DINOv2).
2. **Single-Fold Training Noise:**
   - Currently, Engine 12 and Engine 13 are trained on Fold 0 only. A single fold has sampling variance of $\pm 0.004$ AUC across splits. True Top-1 performance requires completing multi-fold training (Folds 1–4) and ensembling them to suppress variance.
3. **The Gold-58 Sanity Trap:**
   - Gold-58 has only 7 fractures and 9 MCL tears. A single misclassified study shifts Gold-58 class AUC by $\pm 0.035$. Gold-58 can ONLY be used as a sanity check, NEVER as an optimization target.
4. **Public Leaderboard Rounding & Shakeup:**
   - Kaggle rounds public LB to 3 decimal places. Small real gains (+0.0004) are invisible on public LB. Selecting final submissions solely on public LB leads to catastrophic shakeup. Selections must be grounded in multi-fold OOF validation.
5. **Efficiency Prize vs Max AUC Trade-off:**
   - The Efficiency Prize ($18,000) requires total inference $<500\text{s}$. The main competition leaderboard allows up to 9 hours. Constraining our main ensemble to $<500\text{s}$ artificially caps our ceiling.
   - **Solution:** Two-Track Strategy: One submission optimized for the $18,000 Efficiency Prize (<500s), and one unconstrained heavy ensemble for Max AUC.

---

## 3. The 10 Permanent Long-Run Assets to Reach Top 1

To guarantee long-term dominance across any AI agent switch or final-hour scenario, we build and maintain these 10 foundational assets:

1. **Native 384px Spatial Resolution Dominance:**
   - Maintain $\ge 70\%$ ensemble weight on native $\ge 384\times 384$ backbones to preserve crucial micro-structural tear gradients.
2. **Multi-Fold Variance Suppression (5-Fold Averaged Weights):**
   - Train and ensemble all 5 folds of ConvNeXt-Large-384 and Swin-Base-384. A 5-fold ensemble consistently beats a single fold by $+0.003$ to $+0.005$ AUC.
3. **Structural Architectural Orthogonality:**
   - Combine three fundamentally different model families:
     - **Modern ConvNet:** ConvNeXt-Large-384 (Large 7x7 depthwise convolutions, pure inductive bias).
     - **Hierarchical Vision Transformer:** Swin-Base-384 (Shifted window self-attention, long-range global context).
     - **Self-Supervised Medical Semantics:** DINOv2-ViT-L/14 (Frozen patch tokens, zero supervised inductive bias collapse).
4. **Learnable Gated Attention Multiple Instance Learning (Attention-MIL):**
   - Replace naive mean/max pooling with gated multi-head slice attention:
     $$h_{\text{volume}} = \sum_{k=1}^K a_k h_k, \quad a_k = \frac{\exp(w^T (\tanh(V h_k) \odot \sigma(U h_k)))}{\sum_{j=1}^K \exp(w^T (\tanh(V h_j) \odot \sigma(U h_j)))}$$
5. **Multi-Planar Joint Cross-Attention (Sagittal + Coronal + Axial):**
   - Feed all 3 anatomical views into a unified multi-plane transformer encoder to capture spatial interactions (e.g., joint effusion on Axial correlating with ACL tear on Sagittal).
6. **Asymmetric Focal Loss for Rare Classes:**
   - Apply Asymmetric Loss ($\gamma_+ = 0, \gamma_- = 4$) to heavily penalize easy negative examples for Fracture (~2.5% positive) and MCL (~3.2% positive).
7. **Automated AST Guardrail Pipeline (`tools/validate_kernel_guardrail.py`):**
   - Machine-verifies zero fatal assertions, resolution dominance, runtime budgets, and file integrity before every single kernel push.
8. **Decoupled Two-Track Selection Strategy:**
   - **Slot 1 (Safety Floor):** Permanently lock Ref `57253840` (0.958 baseline) to guarantee a top-tier result.
   - **Slot 2 (Ceiling Ambition):** Dedicated to the highest 5-fold OOF ensemble targeting $\ge 0.967$ (Top 10) and $0.969$ (#1).
9. **Rank-Percentile Invariant Blending:**
   - Strictly normalize predictions to empirical percentiles $\frac{\text{rank}(p) - 0.5}{N}$ before linear weighting, ensuring absolute invariance to calibration drift and prevalence shifts.
10. **Living State Telemetry & Cross-Agent Continuity Ledger:**
    - Continuous synchronization via `tools/state_snapshot.py`, `docs/LIVE_STATE.md`, and `docs/LIVE_STEP_LOG.md` ensuring seamless continuity across model architectures and session compactions.
