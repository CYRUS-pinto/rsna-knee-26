# RSNA Knee Abnormality Detection: Evolutionary Roadmap (V1 to V60)

Comprehensive technical log documenting the complete chronological trajectory of our solution in the RSNA Knee Abnormality Detection challenge.

![Evolutionary Trajectory](assets/figure1_evolution_trajectory.png)

## Summary Milestone Table

| Iteration / Candidate | Focus Area & Key Technical Innovations | Public LB | Private LB | Gold-58 Gate | Kaggle Latency (2x T4) | Status / Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Phase 0: Labels** | Qwen-2.5 72B weak label extraction from 4,349 reports; Gold-58 gate | N/A | N/A | 0.9172 CV | N/A | Established foundation |
| **V1 - V5 Prototypes** | Naive 2D CNN backbones; pydicom loading; uncalibrated sigmoids | 0.916 - 0.938 | — | 0.8950 | 510.0 s | Failed: Slices sorted by filename |
| **V6 Dual-Engine** | C++ `_fastread.py` 3D-normal sorting; 140mm crop; 50/50 rank blend | 0.9560 | — | 0.9420 | 480.0 s | **Major Breakthrough (+0.0280 AUC)** |
| **V7 Asymmetric Weights** | Class-dependent 65/35 and 70/30 linear rank blend | 0.9550 | — | 0.9410 | 485.0 s | -0.0010 regression (overfitting) |
| **V8 Horizontal TTA** | Direct horizontal flip test-time augmentation | 0.9550 | — | 0.9405 | 490.0 s | -0.0010 regression (inverted anatomy) |
| **V9 - V11 Heuristics** | Rule-based conditional priors (e.g., if ACL then Contusion floor) | 0.9550 | — | 0.9415 | 485.0 s | -0.0010 regression (non-smooth) |
| **V12 Clean Rank Gate** | Excised heuristic rules (ADR-001/002); paired bootstrap gate | 0.9560 | — | 0.9430 | 475.0 s | Clean baseline restored |
| **V13 - V16 Multi-Scale** | Integrated 224px CoAtNet companion (0.8 center crop) | 0.9570 | — | 0.9480 | 460.0 s | Improved focal meniscal signal |
| **V17 Verified PB** | Full multi-scale Raptor (384 + 224) + Heliosli R6 frozen bundle | **0.9580** | — | 0.9520 | 443.4 s | **Personal Best public benchmark** |
| **V18 - V19 Tuning** | Per-finding weight sweeps (Lateral OA 0.20, Synovitis 0.80) | 0.9580 | — | 0.9525 | 450.0 s | Stabilized specialist weights |
| **V20 Power Rank Blend** | Generalized Power-Mean ($p=1.40$) on ACL, Menisci, Fracture | **0.9580** | **0.9420** | 0.9530 | **469.4 s** (451.3s int) | **Official Verified Private LB Anchor** |
| **V21 - V44 Exploration** | OFL-L self-supervised backbones, Swin-B, MaxViT, DINOv2 | 0.956 - 0.958 | — | 0.9500 | 480 - 520 s | Confirmed CoAtNet + ConvNeXt dominance |
| **V45 Sovereign Master** | Dual-arm anchor + Biomechanical coupling + Fail-soft clamping | **0.9580** | — | 0.9550 | 533.9 s | Peak public LB; high cold-start I/O |
| **V46 - V48 Dilution** | Auxiliary 16-window non-OAI models (`all_vols.npy`) blended at 20-50% | 0.953 - 0.957 | — | 0.9480 | 520.0 s | -0.0050 regression (feature dilution) |
| **V50 Automation Bug** | Agent queried hidden test UIDs in training corpus `all_vols.npy` | **0.4910** | — | 0.7200 | 279.8 s | **Catastrophic failure: Random beta noise** |
| **V51 - V52 Recovery** | Live DICOM streaming + 0.01s path resolver; zero random noise | 0.9560 | — | 0.9530 | 335.0 s | Fully restored pipeline |
| **V53 Real DICOM Apex** | Asynchronous Dual-T4 dispatch; 3-point monotonic slice check | 0.9570 | — | 0.9540 | 510.0 s | Triple rank-quantization identified |
| **V56 Sovereign Reader** | Added 2.5D ConvNeXt Reader; Reader weighted at $W=0.15$ | 0.9570 | — | 0.9540 | 551.7 s | Slight dilution from $W=0.15$ Reader |
| **V57 Directory Bug** | Missing `os.makedirs` before `%%writefile` caused commit crash | N/A | — | N/A | 543.9 s (Crashed) | Dead run |
| **V58 Worker Thrashing** | Raised thread pool from 6 to 12 workers on 4 vCPUs | 0.9570 | — | 0.9540 | 660.9 s (+109s) | Severe GIL & disk cache thrashing |
| **V59 Probit Grandmaster** | Hybrid 50% Gaussian Probit $\Phi^{-1}(p)$ + 50% Continuous Rank | 0.9580 | — | 0.9565 | 530.0 s | Continuous margin preservation |
| **V60 Pure-OAI Master** | Clean V45 + exact Versia_7 reader (0.003-0.035) + 12x12 coupling | **0.9585+** | — | **0.9582** | **452.1 s** | **Final Production Architecture** |

---

## Detailed Chronological Analysis

### Phase 0: Data Foundation & Super-Consensus Derivation
* **Supervision Scarcity**: The RSNA training corpus comprises approximately 4,400 multi-series studies, but only **58 studies** carry expert radiologist annotations ("Gold-58"). The remainder are accompanied by free-text reports in variable languages.
* **Continuous Soft-Label Extraction**: Using Qwen-2.5 72B, free-text reports were converted to continuous probabilistic targets $[0.0, 1.0]$. Ambiguous language ("suspicious for tear", "mild chondromalacia") was preserved as fractional probabilities ($0.75\text{ to }0.85$) rather than thresholded to binary $0/1$.
* **Consensus Aggregation**: Ensembled our extracted labels with public sets (stevenleehans, pilkwang, riadmohamed42) into `SUPER_CONSENSUS_labels.parquet` (4,349 studies, $0.9172$ 5-fold CV).

### Phase 1: Naive Prototypes (V1 to V5)
* **The Slice Sorting Blunder**: Early models loaded DICOMs using alphabetical file paths (`0001.dcm`, `0002.dcm`). In multi-slice acquisitions, scanner slice numbering does not correlate with physical z-depth. Slices were spatially scrambled, preventing models from learning through-plane continuity.
* **Calibration Mismatch**: Ensembling raw sigmoid probabilities from different backbones favored whichever model produced steeper gradients, degrading rank stability.

### Phase 2: Dual-Engine 50/50 Baseline (V6, Public LB 0.9560)
* **Physically Grounded Geometry**: Implemented C++ `_fastread.py` computing slice normal vectors $n = \text{IOP}_{0:2} \times \text{IOP}_{3:5}$ and projected coordinates $d = \text{IPP} \cdot n$. Sorting slices along physical depth produced an immediate **+0.0280 AUC leap**.
* **140 mm Metric Physical Crop**: Normalized pixel spacing to a constant physical field of view.
* **Dual Architecture Fusion**: Blended Raptor-384 CoAtNet with Heliosli ConvNeXt-L R6 at equal 50/50 percentile rank shares.

### Phase 3 & 4: Controlled Experiments & Clean-Up (V7 to V12)
* **Failure of Heuristics**: Heuristic threshold adjustments (e.g., if ACL $> 0.85$ force Contusion $\ge 0.50$) improved apparent validation on small subsets but regressed on the public leaderboard ($0.956 \to 0.955$) due to step discontinuities.
* **Architectural Decisions (ADR-001 / ADR-002)**: Formally excised all hard if/else conditional rules. Instituted the immutable Gold-58 offline paired bootstrap gate ($B=2,000$ iterations).

### Phase 5 & 6: Multi-Scale Power-Rank (V13 to V20, Private LB 0.9420)
* **Dual-Resolution Raptor**: Added a $224 \times 224$ CoAtNet companion focusing on a central $0.80$ crop of the joint space ($35\%$ share of the Raptor branch).
* **Generalized Power-Mean ($p=1.40$)**: Introduced non-linear concordant rank amplification for focal conditions (`ACL`, `Medial Meniscus`, `Lateral Meniscus`, `Fracture`), yielding the verified **0.9420 Private Leaderboard** score in $469.4\text{ s}$ total kernel runtime.

### Phase 7: Sovereign Rank Master (V45, Public LB 0.9580)
* **Fail-Soft Hardening**: Converted fatal runtime exceptions to automated median imputation and added `nan_to_num` sanitization.
* **Smooth Biomechanical Coupling**: Applied continuous linear updates raising ACL scores concordantly with high Contusion and Effusion, and Medial Meniscus scores concordantly with Medial OA.

### Phase 8 & 9: Forensic Diagnostic Period (V46 to V58)
* **V50 Automation Outage (0.4910 LB)**: An automated script runner erroneously looked up hidden test IDs in `all_vols.npy` (which only contained training studies), triggering synthetic random beta fallback code.
* **V53 / V56 Quantization Discovery**: Revealed that calling `rank_matrix()` multiple times sequentially snaps continuous probabilities into discrete step cliffs ($0.3333, 0.6667, 1.0000$).
* **V58 Thread Contention**: Proved that 12 threads on 4 vCPUs caused severe GIL thrashing, slowing inference by $+109\text{ seconds}$.

### Phase 10: Production Pure-OAI & Versia Power Master (V59 to V60)
* **Continuous Margins**: Fully eliminated triple rank-quantization, operating in continuous Power-Probit space.
* **Versia_7 Calibration**: Set Reader weights to exact calibrated ranges ($0.003\text{ to }0.035$, mean $0.016$), eliminating non-OAI feature dilution.
* **12x12 Orthopaedic Coupling**: Applied complete multi-pathway kinetic coupling in a single pass at the final output boundary.
