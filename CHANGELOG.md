# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/). Scores are public LB.

## [V45] Sovereign Rank Master
### Added
- Biomechanical coupling on ACL (Contusion x Effusion) and Medial Meniscus (Medial OA).
- Optional third-arm rank injection (inactive in the shipped run).
### Changed
- Linear rank blend on all findings; per-finding Raptor shares re-tuned.
- Guardrail exceptions replaced by warnings and imputation; non-finite Raptor-384 logits sanitised.
- Output clipped to [0.0001, 0.9999].
### Score
- _Record from Kaggle._ Kaggle run time 533.9 s.

## [V20] Generalized Power-Rank Multi-Scale Blend
### Added
- Power mean (p = 1.4) on ACL, Medial Meniscus, Lateral Meniscus, Fracture.
### Changed
- Built on V17 architecture; Raptor half is 65% 384 px + 35% 224 px; per-finding Raptor/Heliosli weights.
### Score
- **0.942 private.** Kaggle run time 469.4 s.

## [V17] Verified architecture
- **0.958** (personal best), 443.4 s offline. *Notebook not yet in repo.*

## [V13-V16, V18-V19, V21-V44]
- *Not documented in the supplied material. Add entries.*

## [V12] Clean offline
- Removed heuristic priors; restored uniform rank blend. Gold-58 bootstrap gate added.

## [V7-V11] Experiments
- Asymmetric weights (V7), flip TTA (V8), heuristic priors (V9-V11): each 0.955.

## [V6] Dual-engine baseline
- **0.956.** CoAtNet-384 + Heliosli ConvNeXt, 3D-normal slice ordering, uniform 50/50 rank blend.

## [V1-V5] Prototypes
- Below 0.940.
