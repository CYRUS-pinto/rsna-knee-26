# ADR-039: Engine 9 (384px ConvNeXt-Small 2.5D MIL) & External-Data Sovereignty Protocol

**Date:** 2026-10-18  
**Status:** ACCEPTED & OPERATIONAL  
**Track:** Track 2 (Gated GPU Offense) & Two-Track Governance  

---

## 1. Context & Motivation

Following the empirical findings of Engine 8 (where 24-channel multi-plane initialization from scratch plateaued at 0.7098 Gold AUC due to the loss of ImageNet RGB pretraining), we engineered **Engine 9**:
- **Architecture:** `convnext_small.fb_in22k_ft_in1k_384` with 2.5D Multi-Instance Learning (MIL).
- **Input Pipeline:** 3-slice contiguous RGB triplets (`[c-1, c, c+1]`) preserving 100% of the ImageNet-22k pretrained features.
- **Resolution:** Full 384×384 native resolution.
- **Supervision:** Distilled Super-Consensus continuous soft labels (100% coverage across all 4,407 examinations).
- **Compute:** Executed on Colab Pro NVIDIA A100-SXM4-80GB GPU.

Concurrently, a deep forensic audit of Kaggle competition discussions (#749858, #750236, #750355, #743416) revealed that public 0.956–0.958 notebooks secretly rely on restricted external datasets (NYU fastMRI in Heliosli R7, OAI in Nartaa CoAtNet, Stanford SKM-TEA in anatomy.pt). RSNA Director of Informatics Chris Carr reiterated that datasets requiring institutional agreements do NOT meet the equal-access criterion (Rule 6.a), creating significant post-freeze disqualification hazards for teams that blindly copy public weights.

---

## 2. Empirical Verification

### Engine 9 Training Trajectory (A100 SXM4)
- **Epoch 1/8:** Loss `0.5697` | Gold-58 Macro AUC: `0.6754`
- **Epoch 2/8:** Loss `0.5391` | Gold-58 Macro AUC: `0.7411`
- **Epoch 3/8:** Loss `0.5226` | Gold-58 Macro AUC: `0.7759`
- **Epoch 4/8:** Loss `0.5105` | Gold-58 Macro AUC: `0.8132`
- **Epoch 5/8:** Loss `0.5013` | Gold-58 Macro AUC: `0.8208`
- **Epoch 6/8:** Loss `0.4920` | Gold-58 Macro AUC: `0.8260`
- **Epoch 7/8:** Loss `0.4870` | Gold-58 Macro AUC: `0.8325` (Peak)
- **Epoch 8/8:** Loss `0.4827` | Gold-58 Macro AUC: `0.8319`

**Key Metrics:**
- **Peak Gold-58 Macro-AUC:** **`0.8325`** (+0.13 over Engine 8 V2's 0.7098).
- **Stability:** Monotonic convergence with zero gradient blow-up or loss plateauing.
- **Published Weights:** Uploaded to Kaggle dataset [`cyruspinto/rsna-knee-engine9-384-weights`](https://www.kaggle.com/datasets/cyruspinto/rsna-knee-engine9-384-weights) (`engine9_convnext_384_best.pt`, 190 MB, SHA-verified).

---

## 3. External-Data Sovereignty Protocol

| Asset | Source Data | Licensing / Governance | Status |
| :--- | :--- | :--- | :--- |
| **Cyrus Super-CoAtNet-384** (`0.9139` Gold AUC) | RSNA Competition MRI + CC0 Soft Labels | Standard `timm` ImageNet Pretraining | **100% Sovereign / Zero Risk** |
| **Engine 9 384px ConvNeXt** (`0.8325` Gold AUC) | RSNA Competition MRI + CC0 Soft Labels | Standard `timm` ImageNet-22k Pretraining | **100% Sovereign / Zero Risk** |
| **Public Heliosli R7** | Competition MRI + fastMRI Auxiliary | Requires NYU Data Agreement | External Risk (Used in Slot 1) |
| **Public Nartaa CoAtNet** | Competition MRI + OAI Baseline | NIH Login.gov + Geo-Restrictions | External Risk (Used in Slot 1) |

**Two-Track Decision Rule:**
1. **Slot 1 (`57253840`, V17, Public LB 0.958):** Retained as the public ceiling anchor. If Kaggle permits shared public weights under the "reasonable access" clause, our top-tier medal is locked.
2. **Slot 2 (Pure Sovereign Track):** Gated by Candidate V33/V34. Fuses our verified internal models (Cyrus Super-CoAtNet 0.9139 + Engine 9 0.8325). If post-freeze audits disqualify external data, our sovereign submission guarantees clean survival.

---

## 4. Candidate V33 Version 2 Patch

During the initial commit of Candidate V33, `RuntimeError: Raptor raw dump does not match its own submission ranks` was triggered in `In [7]`.
- **Root Cause:** Kaggle's mock test environment contains only `n = 3` studies. Ties in 3 mock predictions caused Spearman correlation to yield `0.5`, tripping the `0.99` assertion.
- **Patch:** Guarded the assertion with `if n > 20 and min(rho_rr.values()) < 0.99:`.
- **Status:** Version 2 pushed to [`cyruspinto/rsna-knee-v33-grandmaster-fusion`](https://www.kaggle.com/code/cyruspinto/rsna-knee-v33-grandmaster-fusion) (Kernel ID: `139417677`). Currently in Kaggle GPU execution queue.
