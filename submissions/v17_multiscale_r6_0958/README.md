# Submission Candidate V17: Multi-Scale Raptor (384 + 224) + ConvNeXt R6 Flip TTA (0.9576+ Baseline)

## 1. Specification & Rationale
- **Target Public LB:** **0.9576 – 0.958+**
- **Target Runtime:** **~425s** (Fully compliant with the $18,000 Efficiency Prize Pool ceiling of 500s)
- **Architectures:**
  1. **Stage 1A (CoAtNet-384 SWA):** Nartaa's public 384px full-field CoAtNet trained on JEV soft labels (Weight: 0.65 of Raptor half).
  2. **Stage 1B (CoAtNet-224 SWA):** Nartaa's 224px center-crop (0.8 field) CoAtNet (Weight: 0.35 of Raptor half).
  3. **Stage 2 (ConvNeXt-S6 R6 with Flip TTA):** Heliosli's 4-fold bundle `fe055406-w1` with bilateral anatomical reflection TTA and dedicated volumetric fluid stackers for Effusion & Baker's Cyst.
  4. **Stage 3 (Scale-Invariant Calibrated Rank Fusion):**
     - Base: 50% Raptor Half + 50% ConvNeXt R6 Half.
     - Calibrated finding overrides based on proven validation strength:
       - `Lateral OA`: 0.20 Raptor / 0.80 ConvNeXt (ConvNeXt coronal extent heads dominate).
       - `PF OA`: 0.60 Raptor / 0.40 ConvNeXt (Raptor axial windows dominate).
       - `Synovitis`: 0.80 Raptor / 0.20 ConvNeXt (Raptor soft-tissue contrast).
       - `Baker's`: 0.35 Raptor / 0.65 ConvNeXt (ConvNeXt dedicated Baker's stacker).
       - `Medial OA`, `ACL`, `MCL`, `Menisci`, `Contusion`, `Fracture`: 0.50 / 0.50 uniform rank fusion.

## 2. Safety & Verification Guardrails
- **Zero Fallback Failure Tolerance:** Stops if >3% of Raptor studies fail or >50% of ConvNeXt studies fail.
- **Monotonic Percentile Ranking:** Completely scale-invariant (eliminates raw probability calibration shifts).
- **Schema Assertions:** Wide-format `(N, 13)` (`StudyInstanceUID` + 12 finding probabilities in $[0.0, 1.0]$).
