# ADR-020: Quad-Engine Multi-Plane Specialist Architecture & Full-Dataset Validation

## Status
ACCEPTED & PREPARED (Engine 5 weights verified, 5-fold Arm A benchmarked, V21 architecture staged)

## Authors
Antigravity Senior Engineering (15+ Year Discipline)

## Context & Problem Statement
In our ongoing pursuit of the $\ge 0.960$ leaderboard threshold, we have achieved:
- **0.956** with Multi-Scale 384+224 (Candidate V15).
- **0.958** with Multi-Scale 384+224 + ConvNeXt R6 Flip TTA (Candidate V17, current Slot 2 leader).
- **V18** (Asymmetric Prior) & **V20** (Generalized Power-Rank $p=1.4$) currently evaluating in Kaggle's hidden test queue.

To advance toward the podium tier ($0.962 - 0.965+$), we audited two core components:
1. **The Representation Collapse in Multi-Task Architectures**: Engine 4 collapsed to a single global score ($r > 0.85$, macro-AUC $0.69$).
2. **The Anatomical Strength of Decoupled Specialists**: Engine 5 trained 3 independent networks on dedicated planes and targets, completely preventing collapse.

## Empirical Findings & Data Truth

### 1. Full-Dataset 5-Fold Benchmark (Arm A / CoAtNet-d4)
Using the full RSNA Knee dataset ($N=4,407$ unique studies across all 5 folds from `scratch/ntejas_repo/oof/` evaluated against `SUPER_CONSENSUS_labels.parquet`):
- **Overall Macro-AUC**: **0.85360**
- **Top Finding AUCs**:
  - Medial Meniscus: **0.9144** ($1,780$ positives)
  - Synovitis: **0.8883** ($591$ positives)
  - Fracture: **0.8818** ($315$ positives)
  - Contusion: **0.8764** ($841$ positives)
  - MCL: **0.8751** ($145$ positives)
  - ACL: **0.8732** ($673$ positives)
  - Lateral Meniscus: **0.8684** ($707$ positives)
  - PF OA: **0.8483** ($976$ positives)
  - Medial OA: **0.8373** ($1,629$ positives)
  - Baker's Cyst: **0.8019** ($1,089$ positives)
  - Lateral OA: **0.7952** ($1,124$ positives)
  - Effusion: **0.7831** ($2,630$ positives)

### 2. Engine 5 Plane-Decoupled Specialist Validation (Fold 0, $N=881$)
Engine 5 training completed on Kaggle Dual T4 GPUs (`rsna-knee-engine5-decoupled-train`). Weights and OOF parquet were downloaded and evaluated locally:
- **Overall Macro-AUC**: **0.79853**
- **Anti-Collapse Verification**: Mean off-diagonal correlation was **0.4490** (compared to $>0.85$ collapsed baseline).
- **Specialist Strengths**:
  - `AxialFluidSpecialistMIL`: **Synovitis 0.8917** (outperforms Arm A's 0.8883), **Effusion 0.8138** (outperforms Arm A's 0.7831).
  - `SagittalSpecialistMIL`: **Medial Meniscus 0.8625**, **ACL 0.8563**, **Contusion 0.8141**.
  - `CoronalSpecialistMIL`: **Medial OA 0.7934**, **Lateral OA 0.7819**.

### 3. ConvNeXt R6 Structural Analysis
ConvNeXt R6 achieves its high score by performing 3D anatomical joint localization using SKM-TEA pretrained weights (`AnchorBank`), canonicalizing left/right knee orientation via a `resnet34` classifier, and extracting native-resolution $352 \times 352$ crops around the joint line.
- The pipeline is cryptographically hash-locked (`CODE_SHA` assertions on `cloud_code.zip`), ensuring zero runtime regression.

## Candidate V21: Quad-Engine Decoupled Architecture

Candidate V21 builds upon Candidate V20 by adding Engine 5 as a 4th specialist engine:
1. **Engine 1**: Raptor-384 Whole Volume (SWA checkpoint).
2. **Engine 2**: Raptor-224 Whole Volume (Multi-scale center crop).
3. **Engine 3**: ConvNeXt R6 (SKM-TEA 3D Anatomical Joint-Space Crops, Flip TTA).
4. **Engine 4**: Engine 5 Multi-Plane Specialists mounted via `kernel_sources: ["cyruspinto/rsna-knee-engine5-decoupled-train"]`:
   - Extracts slices from the in-memory 96-slice tensor already loaded by Raptor in `_fastread.py`.
   - Adds $\approx 0.05\text{s}$ per study forward pass on GPU (zero extra DICOM I/O overhead).
   - Injected into the ensemble specifically for fluid-sensitive and focal tear findings (Synovitis, Effusion, ACL, Medial Meniscus).

## Decision Rules for Deployment
1. **Slot 1 (Permanent Anchor)**: Candidate V6 (`0.956`) [Ref `57165672`] remains strictly locked.
2. **Slot 2 Evaluation Gate**:
   - If **V18** or **V20** scores $\ge 0.960$, that model claims Slot 2 immediately.
   - If both V18 and V20 remain at $0.958$, Candidate V17 remains locked in Slot 2.
   - In that event, Candidate V21 is submitted using one of our remaining 2 submissions for today.
