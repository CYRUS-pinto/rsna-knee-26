# ADR-038: Engine 8 Multi-Plane ConvNeXt Architecture and Candidate V33 Tri-Hybrid Rank Fusion

**Date:** 2026-10-18  
**Status:** ACCEPTED & IN EXECUTION  
**Context:** Final Sprint (~30 Hours to Code Freeze). Public LB Baseline: **0.958 (Rank 463)**. Target: **Top 1 / 0.965–0.969**.

---

## 1. Autopsy of Engine 7 (DINOv2 Distilled ViT) & Gate Enforcement
- **Execution:** Pushed as `cyruspinto/rsna-knee-engine7-dinov2-distilled-train` (Version 3) on Kaggle GPU.
- **Result:** Completed 8 epochs. Loss flatlined (`0.5729` $\rightarrow$ `0.5699`). Peak Gold-58 Macro-AUC: **0.5607**.
- **Root Cause:** Pure Vision Transformers (ViT-S/14) lack 2D spatial inductive bias. Slicing 16 volumetric windows generated $16 \times 256 = 4,096$ patch tokens per study. Without medical pretraining, 8 epochs over 3,480 soft-labeled studies is mathematically under-determined for cross-attention localization.
- **Governance Action:** **Gate 1 REJECTED (AUC 0.5607 < 0.9000)**.
  - Zero submissions burned on Kaggle.
  - Zero Colab compute units wasted.
  - Defense Track (Slot 1: 0.958, Slot 2: 0.956) remains 100% intact.

---

## 2. Discovery & Audit of Cyrus Super-CoAtNet-384
- **Asset:** `cyruspinto/rsna-knee-colab-super-weights` (`colab_super_coatnet_0917_best.pt`, SHA256: `b289852d1eeb48c4b9c588bd52148f429166c4b1a483b6c261c0d992ab063630`).
- **Architecture:** `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (Identical parameter shapes to Raptor-384, but supervised on continuous consensus soft labels).
- **Gold-58 Radiologist Audit:**
  - **Macro-AUC:** **0.9139**
  - **Synovitis:** **0.8554** (Highest synovitis AUC recorded in the entire codebase)
  - **Medial Meniscus:** **0.9627**
  - **Effusion:** **0.9143**
  - **ACL:** **0.9767**
  - **Medial OA:** **0.9876**
- **Action:** Immediately drafted as the 3rd orthogonal arm for Candidate V33.

---

## 3. Candidate V33 Grandmaster Rank Fusion
- **Kernel:** `cyruspinto/rsna-knee-v33-grandmaster-fusion` (Kernel ID: `139417677`, pushed and queued on Kaggle GPU).
- **Architecture:**
  1. **Stage 1A (CoAtNet-384 Dual-Arm):** Raptor SWA (0.65) + Cyrus Super-CoAtNet (0.35). Evaluated within a single DICOM decode pass.
  2. **Stage 1B (CoAtNet-224):** Nartaa 224-crop SWA (0.30 share of Raptor arm).
  3. **Stage 2 (ConvNeXt-R7 Stride-2):** 4-fold ensemble with Stride-2 window pruning cutting inference windows by 50%.
  4. **Stage 3 (Quantile Fusion):** Scale-invariant percentile rank quantization across all 12 pathologies.
- **Displacement Rule:**
  - Public LB $\ge 0.9610 \rightarrow$ Displace Slot 2 (V6 $\rightarrow$ V33).
  - Public LB $< 0.9610 \rightarrow$ Retain Slot 2 (V6: 0.956 locked).
  - Slot 1 (V17: 0.958) is never modified.

---

## 4. Engine 8: 24-Channel Multi-Plane ConvNeXt-Tiny (Colab A100)
- **Clinical Motivation:** Sagittal-only models are physically blind to joint-space narrowing on Medial OA, Lateral OA, and PF OA (accounting for 25% of the competition metric).
- **Architecture:** Stacks 12 Sagittal slices + 12 Coronal slices into a 24-channel 2D input. Modern 7x7 depthwise convolutions in ConvNeXt-Tiny jointly learn depth and cross-plane boundaries.
- **Hardware Profile:** Executing on Colab Pro NVIDIA A100 SXM4 (80GB VRAM, 167GB System RAM).
  - `mmap_mode='r'` leverages 167 GB host RAM for zero-latency in-memory slice lookups.
  - PyTorch AMP Tensor Core acceleration.
  - Expected runtime: ~10 minutes for 10 epochs.
- **Output:** Automatically validates Gold-58 AUC and pushes `engine8_convnext_best.pt` to Kaggle Dataset `cyruspinto/rsna-knee-engine8-multiplane-weights`.
