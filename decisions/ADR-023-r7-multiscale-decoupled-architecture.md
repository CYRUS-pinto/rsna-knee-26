# ADR-023: High-Resolution 384px R7 Integration & Decoupled Multi-Scale Architecture

## Status
UNVERIFIED (Candidate V23 kernel pushed and running; runtime and decoupling weights pending empirical measurement)

## Context & Forensic Root Cause
1. **The 0.958 Plateau:** Candidates V15, V17, V18, V20, and V22 all hovered between `0.956` and `0.958` because they were re-weighting the same two public feature extractors:
   - Raptor CoAtNet-2 (384px + 224px, 96-slice multi-view).
   - Heliosli ConvNeXt-small R6 (bundle `fe055406`).
2. **The V21 Collapse (0.949):**
   - V21 attempted a non-linear power-rank focal amplification ($p=1.4$) on top of discordant predictions where Raptor and ConvNeXt had negative rank correlation on `Lateral Meniscus` and `Medial OA`.
   - Exponentiating discordant rank distributions amplified outliers across the distribution tail, collapsing public score to `0.949`.
3. **The High-Resolution R7 Bundle:**
   - Heliosli published R7 (bundle `8c7b666e27dc0e8f`), which runs ConvNeXt-small at 384px (crop 352px) with updated synovial decomposition (`syn_beta`) and rare class boosting.
   - Note on R6 crop: Prior documentation claimed R6 was 208px; the V17 receipt shows crop 352px was already active for R6. R7's difference is in the retrained encoder weights, synovial decomposition, and rare class weighting.

## Architecture of Candidate V23
Candidate V23 creates a multi-scale, dual-backbone ensemble:
1. **Stage 1 (Full-Field Multi-Slice Context):**
   - Raptor CoAtNet-2 (384px + 224px, 96-slice span).
2. **Stage 2 (Joint-Anchored High-Resolution Crops):**
   - Heliosli ConvNeXt-small R7 (384px, 4 folds, SKM-TEA knee-joint segmentation, synovial decomposition).
3. **Stage 3 (Decoupled Compartmental Blend):**
   - Per-finding authority:
     - `Lateral Meniscus`: R7 85% / Raptor 15% (CLAIMED: Pending cross_arm_check verification)
     - `Medial OA`: R7 80% / Raptor 20% (CLAIMED: Pending cross_arm_check verification)
     - `Lateral OA`: R7 80% / Raptor 20%
     - `Synovitis`: Raptor 80% / R7 20%
     - `ACL`: Raptor 70% / R7 30%
     - `MCL`: Raptor 65% / R7 35%
     - `PF OA`: Raptor 60% / R7 40%
     - `Baker's`: R7 65% / Raptor 35%
     - `Medial Meniscus`, `Contusion`, `Effusion`, `Fracture`: 50/50 rank blend.

## Latency & Safety Verification
- **Commit Run Timing (MEASURED):** On the 3-study sample test set, Raptor stages take ~75s (R384 ~51s, R224 ~24s). ConvNeXt stage takes ~350s. Total commit time is ~450–500s (~150s per study on the 3-study sample).
- **Hidden Test Latency (ESTIMATED):** The full hidden test set is ~1,000–1,300 studies. If per-study inference on hidden test equals ~12–15s per study (with fast read and pipelining), full runtime is estimated at ~3.5–5.5 hours, within Kaggle's 9.0-hour hard limit. Real hidden test runtime must be verified from Kaggle Submissions page.
- **Safety Invariant:** Slot 1 (`0.956`, Ref `57165672`) remains locked. Slot 2 (`0.958`, Ref `57253840`) is only replaced if Candidate V23 achieves $\ge 0.960$.
