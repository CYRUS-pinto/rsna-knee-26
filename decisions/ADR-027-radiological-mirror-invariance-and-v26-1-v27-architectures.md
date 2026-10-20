# ADR-027: Radiological Mirror Invariance and Candidate V26.1 / V27 Architectures

**Status:** ACCEPTED  
**Date:** 2026-10-17  
**Author:** Cyrus Pinto & Pair Engineering  
**Context:** RSNA Knee Abnormality Detection (5 days remaining)

---

## 1. Radiological Anatomy & Mirror Invariance Proof

A critical question arose regarding Test-Time Augmentation (TTA) via horizontal flipping:
> *If an MRI slice is horizontally mirrored, does the Medial compartment swap to Lateral, requiring target swapping during inference?*

### The Anatomical Proof
1. **Body Planes & Laterality:**
   - In human anatomy, "Medial" is defined as *towards the body midline*, and "Lateral" is defined as *away from the body midline*.
   - In both Left and Right knees, the **fibula is always lateral** (outer), while the tibial plateau is medial (inner).
2. **Mirror Reflection across the Sagittal Plane:**
   - Reflecting a Right knee across the sagittal plane produces a geometrically and anatomically valid **Left knee**.
   - On that reflected knee, the fibula remains lateral, and the tibial side remains medial.
   - A tear on the medial meniscus of the original knee remains on the medial meniscus of the reflected knee.
3. **Implication for Neural Networks:**
   - CoAtNet and ConvNeXt models are trained on whole-knee volumetric slices comprising ~50% Left knees and ~50% Right knees without artificial laterality canonicalization.
   - The neural network recognizes internal anatomical landmarks (fibular head, condylar geometry) to identify compartments.
   - Consequently, when the knee is mirrored, the model's prediction for Medial Meniscus remains Medial Meniscus, and Lateral remains Lateral.
   - **Conclusion:** Direct averaging $0.5 \times (p_{\text{plain}} + p_{\text{mirrored}})$ preserves radiological truth. Naive column swapping ($Medial \leftrightarrow Lateral$) would corrupt predictions by inverting compartmental anatomy.

---

## 2. Candidate Architecture Specifications

### Candidate V26.1-Fast ($18,000 Efficiency Prize Champion)
- **Directory:** `submissions/v26_1_fast_efficiency_master`
- **Architecture:** Pure MultiScale Raptor (384px SWA + 224px SWA) using the C-struct `_fastread.py` DICOM loader.
- **Capacity Re-Weighting:** Re-weights the multiscale blend from $65/35$ to **$75/25$ ($A_{224} = 0.25$)**, giving higher authority to the superior 384px model ($0.949$ standalone).
- **Runtime Profile:** $\sim 5\text{--}6\text{ minutes}$ on Kaggle dual T4s.
- **Efficiency Metric:** Targets Top 10–15 on the official 5,985-team Efficiency Leaderboard.

### Candidate V27 (Main Leaderboard Capacity Master)
- **Directory:** `submissions/v27_r7_multiscale_capacity_master`
- **Architecture:** Dual-Arm Master Ensemble:
  1. Raptor MultiScale (384px + 224px with $75/25$ capacity weighting).
  2. ConvNeXt R7 (384px knee-joint crop, 4 folds) with **Flip TTA enabled** (`FLIP_TTA = True`).
  3. ADR-024 Uniform Master Blend ($0.50 / 0.50$ across all 12 pathologies).
- **Runtime Profile:** $\sim 3.5\text{ hours}$ on Kaggle dual T4s.
- **Main Leaderboard Target:** Break through $0.958$ towards $0.960\text{--}0.962$.

---

## 3. Submission Strategy Matrix

1. **Submission Quota Status:** 4 of 5 used today. Exactly 1 submission remains for today's 24-hour cycle.
2. **Pending Pipeline:**
   - Ref `57298964` (Candidate V24): 384px R7 + Raptor Dual-Crop + ADR-024 Uniform Master Blend.
   - Ref `57301656` (Candidate V25): Candidate V24 + ConvNeXt R7 Flip TTA.
3. **Deployment Gate:**
   - Once V24 and V25 scores land, evaluate whether ConvNeXt Flip TTA moved the needle on Public LB.
   - If V25 scores $\ge 0.960 \implies$ select for Slot 2.
   - Dispatch the 5th submission to either lock down the Efficiency Prize (Candidate V26.1-Fast) or reinforce the Main Track (Candidate V27).
