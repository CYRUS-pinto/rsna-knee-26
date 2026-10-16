# ADR-018: Competition Data Landscape, Prevalence Shift Vulnerability, and the Rank-Quantile Grandmaster Blueprint

**Status:** ACCEPTED  
**Date:** 2026-10-16  
**Author:** Principal ML Systems Engineer & Kaggle Grandmaster Team  
**Context:** RSNA Knee Abnormality Detection (7 Days Remaining to Final Deadline)

---

## 1. Context & Competition Data Deconstruction

Analysis of the official dataset specification (819,640 files, 569.76 GB) and Kaggle discussion threads (`748111` "Single sub 0.960! Wow!", `749858` "What external data is inside the public 0.956/0.957 blends", and `743416` "External data disclosure"):

### 1.1 The Official Prevalence Shift Notice
The competition organizers explicitly disclosed:
> *"Although efforts have been made to ensure each abnormality is represented in each dataset, the prevalence of abnormalities is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets."*

In macro-averaged ROC-AUC evaluation, this warning is a critical structural risk:
1. When blending probabilities $P_A$ and $P_B$ from models trained on differing distributions (e.g., OAI pretraining vs. RSNA report pseudo-labels), naive arithmetic averaging ($0.5 P_A + 0.5 P_B$) skews the joint decision boundary if the calibration curves deviate.
2. If Private LB prevalence shifts from Public LB prevalence, models with uncalibrated high-confidence sigmoid tails experience severe rank inversion.
3. **The Solution:** Rank-Quantile transformation (empirical CDF normalization) before ensembling. Because ROC-AUC evaluates ordinal ranking rather than absolute log-loss, rank-averaging preserves the true discriminatory power independent of class prevalence.

### 1.2 Multi-Sequence Physics & Anatomical Planes
The dataset provides three essential metadata flags per series:
1. `Fluid_Sensitive` (T2/PD/STIR): Essential for identifying `Effusion`, `Synovitis`, `Baker's cyst`, and acute ligament tears where edema is hyperintense.
2. `Fat_Suppression`: Crucial for detecting `Bone Contusion` and subchondral fractures by suppressing bright bone marrow fat signal.
3. `Anatomical_Plane`:
   - **Sagittal**: Primary plane for `ACL` and `Meniscus` root/body anterior-posterior morphology.
   - **Coronal**: Primary plane for `MCL`, `Medial OA`, `Lateral OA` joint space narrowing, and collateral meniscal tears.
   - **Axial**: Primary plane for `PF OA` (patellofemoral cartilage wear), `Effusion`, and `Baker's cyst` (popliteal fossa fluid collection).

---

## 2. Competitive Landscape: The 0.960 Single Model & External Data Controversy

### 2.1 How 0.960 Was Achieved by Single Submissions (Discussion 748111)
Optimo and top single-model competitors broke through the 0.956 wall not by training a basic 2D CNN on noisy report labels, but by:
1. **Volumetric Feature Attention**: Cross-plane feature aggregators that query coronal and sagittal representations conditioned on `Fluid_Sensitive` flags.
2. **Auxiliary Medical Pretraining**: Pre-training backbones on curated knee MRI datasets (OAI, fastMRI, SKM-TEA) with segmentation/mask objectives before fine-tuning on RSNA gold labels.

### 2.2 External Data Compliance (Rule 2.6.a/b & Discussion 743416)
Our operational rule is strict adherence to competition-eligible weights. The frozen ConvNeXt R6/R7 and Raptor weights utilized in V17 and V18 are fully pre-compiled and self-contained within Kaggle datasets, ensuring full reproducibility and compliance.

---

## 3. The Grandmaster Endgame Plan (Final 7 Days)

### Principle: Zero Wasted Compute
- **Prohibited:** Re-training 570 GB raw DICOM pipelines from scratch without verified clusters (guaranteed failure mode with 7 days remaining).
- **Mandated:**
  1. Complete evaluation of live submitted candidates:
     - **V17** (`57253840`): Multi-scale (384 + 224) + ConvNeXt R6 Flip TTA (443.4s).
     - **V18** (`57260494`): Asymmetric finding-specific weights + soft synovitis prior (474.1s).
  2. Prototype **V19**: The Rank-Quantile Preserving Ensemble with Clinical Co-Occurrence Priors.
  3. Ensure all submissions remain strictly under the **500s threshold** to secure the **$18,000 Efficiency Prize**.
  4. Maintain **Slot 1 permanently locked to V6 (0.956)** as the risk-mitigation floor.

---

## 4. Verification & Audit Trail
- Verified against competition specifications: 12 binary findings, macro ROC-AUC metric, 1,300 hidden test cases.
- Submission integrity confirmed via [`tools/verify_submission_integrity.py`](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/tools/verify_submission_integrity.py).
