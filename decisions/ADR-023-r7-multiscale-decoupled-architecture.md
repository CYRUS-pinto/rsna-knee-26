# ADR-023: High-Resolution 384px R7 Integration & Decoupled Multi-Scale Architecture

## Status
Accepted / Deployed as Candidate V23 (`cyruspinto/rsna-knee-v23-r7-multiscale-decoupled-blend`)

## Context & Forensic Root Cause
1. **The 0.958 Plateau:** Candidates V15, V17, V18, V20, and V22 all hovered between `0.956` and `0.958` because they were all re-weighting the exact same two feature extractors:
   - Raptor CoAtNet-2 (384px + 224px, 96-slice multi-view).
   - Heliosli ConvNeXt-small R6 (bundle `fe055406`, 224px downsampled input).
2. **The V21 Collapse (0.949):**
   - V21 attempted a non-linear power-rank focal amplification ($p=1.4$) on top of discordant predictions where Raptor and ConvNeXt had a $-0.5$ Spearman correlation (`Lateral Meniscus` and `Medial OA`).
   - Taking $x^{1.4}$ on divergent predictions severely degraded the rank order across the distribution tail, collapsing the public LB score to `0.949`.
3. **The High-Resolution Opportunity (Heliosli R7):**
   - Heliosli published R7 (bundle `8c7b666e27dc0e8f`), which upgrades the ConvNeXt-small feature extractor to native **384px / crop 352px** (nearly $3\times$ higher pixel area than R6's 208px crop).
   - In knee MRI, subtle pathology such as meniscal root tears, osteochondral micro-fractures, and focal cartilage defects are lost in 208px crops but preserved at 352px.
   - R7 also incorporates synovial decomposition (`syn_beta`) and rare class boosting.

## Architecture of Candidate V23
Candidate V23 creates a true multi-scale, dual-backbone, high-resolution ensemble:
1. **Stage 1 (Full-Field Multi-Slice Context):**
   - Raptor CoAtNet-2 (384px + 224px, 96-slice span, anatomical mirror TTA).
   - Authority on cruciate and collateral ligaments (`ACL`, `MCL`).
2. **Stage 2 (Joint-Anchored High-Resolution Crops):**
   - Heliosli ConvNeXt-small R7 (384px, 4 folds, SKM-TEA knee-joint segmentation, synovial decomposition).
   - Authority on compartmental cartilage and menisci (`Lateral Meniscus`, `Medial OA`, `Lateral OA`, `Baker's`).
3. **Stage 3 (ADR-022 Decoupled Compartmental Blend):**
   - Resolves the $-0.5$ Spearman inversion by assigning per-finding authority:
     - `Lateral Meniscus`: R7 85% / Raptor 15%
     - `Medial OA`: R7 80% / Raptor 20%
     - `Lateral OA`: R7 80% / Raptor 20%
     - `Synovitis`: Raptor 80% / R7 20%
     - `ACL`: Raptor 70% / R7 30%
     - `MCL`: Raptor 65% / R7 35%
     - `PF OA`: Raptor 60% / R7 40%
     - `Baker's`: R7 65% / Raptor 35%
     - `Medial Meniscus`, `Contusion`, `Effusion`, `Fracture`: Balanced 50/50 rank blend.

## Latency & Safety Verification
- **Runtime Budget:** Native inference runs in ~450 seconds on interactive test set (~15s per study).
- **Hidden Test Projection:** ~4.1 hours on 1,000 hidden test studies, well under Kaggle's 9.0-hour hard cutoff.
- **Safety Invariant:** Slot 1 (`0.956`, Ref `57165672`) remains locked. Slot 2 (`0.958`, Ref `57253840`) is only replaced if Candidate V23 achieves $\ge 0.960$.
