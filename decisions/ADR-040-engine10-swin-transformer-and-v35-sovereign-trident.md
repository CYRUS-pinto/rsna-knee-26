# ADR-040: Engine 10 (Swin-Base-384 Focal MIL) & Candidate V35 Sovereign Trident Master

**Date:** 2026-10-19  
**Status:** ACCEPTED & OPERATIONAL  
**Track:** Track 2 (Gated Sovereign Offense) & Two-Track Governance  

---

## 1. Context & Motivation

To surpass the public ensemble ceiling (0.958) and protect against private leaderboard shakeup, we established the **Trident Sovereign Ensemble Architecture**:
1. **Hybrid Conv-Attention Arm:** Cyrus Super-CoAtNet-384 (Gold AUC `0.9139`, Meniscus `0.9627`).
2. **Pure Convolutional Arm:** Engine 9 ConvNeXt-Small 384px (Gold AUC `0.8325`, Medial OA `0.9845`, Effusion `0.9168`).
3. **Pure Hierarchical Vision Transformer Arm:** Engine 10 Swin-Base-384 with shifted-window non-convolutional self-attention.

---

## 2. Engine 10 Empirical Results (Colab Pro A100 GPU)

Engine 10 (`swin_base_patch4_window12_384.ms_in22k_ft_in1k`) was trained with 2.5D Multi-Instance Learning and Asymmetric Focal Loss ($\gamma = 1.5$) targeting hard-negative soft-tissue pathologies.

### Training Trajectory:
- **Epoch 1/8:** Loss `0.2056` | Gold-58 Macro AUC: `0.5809`
- **Epoch 2/8:** Loss `0.2031` | Gold-58 Macro AUC: `0.6126`
- **Epoch 3/8:** Loss `0.2004` | Gold-58 Macro AUC: `0.6138`
- **Epoch 4/8:** Loss `0.1970` | Gold-58 Macro AUC: `0.6362`
- **Epoch 5/8:** Loss `0.1946` | Gold-58 Macro AUC: `0.6530`
- **Epoch 6/8:** Loss `0.1912` | Gold-58 Macro AUC: `0.7017`
- **Epoch 7/8:** Loss `0.1886` | Gold-58 Macro AUC: `0.7311`
- **Epoch 8/8:** Loss `0.1867` | Gold-58 Macro AUC: **`0.7372`** (Peak)

### Per-Pathology Breakdown (Top Solved Classes):
- **Medial Osteoarthritis:** `0.9256`
- **Joint Effusion:** `0.9491`
- **Lateral Osteoarthritis:** `0.8453`
- **Synovitis:** `0.7999`
- **Patellofemoral OA:** `0.7696`
- **Lateral Meniscus:** `0.7615`
- **Baker's Cyst:** `0.6775`
- **Fracture:** `0.6569`
- **Bone Contusion:** `0.6174`

**Published Weights:** [`cyruspinto/rsna-knee-engine10-swin-weights`](https://www.kaggle.com/datasets/cyruspinto/rsna-knee-engine10-swin-weights) (`engine10_swin_384_best.pt`, 332 MB).

---

## 3. Candidate V35 Architecture (Sovereign Trident Master)

Candidate V35 integrates 5 distinct engines on dual Tesla T4 GPUs:
- **Arm 1:** Raptor CoAtNet-384 + Cyrus Super-CoAtNet-384 (`0.9139`)
- **Arm 2:** Nartaa CoAtNet-224 SWA (0.8 center crop mirror)
- **Arm 3:** Heliosli ConvNeXt-Small R7 (4-fold ensemble + Stride-2)
- **Arm 4:** Engine 9 ConvNeXt-Small 384px (`0.8325`, Gated Attention MIL Head)
- **Arm 5:** Engine 10 Swin-Base-384 (`0.7372`, Shifted Window Attention MIL Head)

**Pushed Kernel:** [`cyruspinto/rsna-knee-v35-sovereign-trident-master`](https://www.kaggle.com/code/cyruspinto/rsna-knee-v35-sovereign-trident-master).

---

## 4. Colab Pro Pipeline: Engine 9 Fold 1

To eliminate single-fold validation variance and build a robust 2-Fold Sovereign Ensemble:
- **Notebook Generated:** `colab_engine9_fold1_master.ipynb`
- **Dataset:** Disjoint 80% split (Fold 1) of 4,407 examinations.
- **Estimated Run Time:** ~25 minutes on A100 GPU (~3 compute units).
- **Target:** Publish `cyruspinto/rsna-knee-engine9-fold1-weights`.
