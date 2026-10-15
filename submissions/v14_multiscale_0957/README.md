# Submission Candidate V14: Multi-Scale Raptor (384px + 224px) + ConvNeXt R7

## 1. Overview
- **Kaggle Kernel:** `cyruspinto/rsna-knee-multiscale-0957` (Version 1)
- **Target Public LB:** **0.957** (Reproducing the highest public benchmark)
- **Architecture:** Multi-Resolution CoAtNet SWA + 4-fold ConvNeXt R7
  - **Stage 1A:** CoAtNet-384 SWA (`raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`) -> Weight $0.65$
  - **Stage 1B:** CoAtNet-224 SWA (`raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt`) -> Weight $0.35$
  - **Stage 2:** ConvNeXt-S6 R7 (`ARM = 'E_lrprior'`, 4 folds, 352x352 with OA extent heads)
  - **Stage 3:** Scale-Invariant Non-Parametric Percentile Rank Fusion:
    $$r_{\text{raptor}} = 0.65 \cdot \text{Rank}(S_{384}) + 0.35 \cdot \text{Rank}(S_{224})$$
    $$R_{\text{final}} = 0.50 \cdot r_{\text{raptor}} + 0.50 \cdot \text{Rank}(S_{\text{ours}})$$

## 2. Statistical Mechanics & Efficiency Defense
1. **Multi-Scale Spatial Hierarchy:**
   - 384px preserves high-frequency edge gradients for acute tears (ACL, MCL, meniscal rim).
   - 224px downsampling expands receptive field for diffuse compartment degeneration (Effusion, Baker's Cyst, Patellofemoral OA).
2. **Efficiency Prize Envelope:**
   - Standalone Raptor 384: ~78.8s
   - Raptor 224 incremental pass: ~32.0s
   - ConvNeXt R7 stage: ~359.7s
   - **Total Estimated Latency:** **~470.5s** (Under the 500s threshold for the $18,000 Efficiency Prize pool).
3. **Failsafe Design:**
   - Non-raising correlation guard (`min_rho >= 0.30`).
   - 0.50 neutral fallback on unreadable/truncated DICOM streams.
