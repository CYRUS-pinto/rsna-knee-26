# RSNA Knee Abnormality Detection: Leaderboard Intelligence & Competitive Dynamics

Strategic post-mortem analyzing the competitive dynamics of the RSNA Knee Abnormality Detection leaderboard, the 0.9580 tie cluster, and the Gold-58 validation dynamics.

## The 0.9580 Public Leaderboard Tie Cluster

At the close of the competition, over 800 teams were tied at **0.9580 Public Leaderboard AUC** (tied cohort around Rank #859):

```
+------------------------------------------------------------------------------------+
| THE 0.9580 LEADERBOARD DYNAMICS                                                    |
+-------------------+----------------------------------------------------------------+
| Public Baseline   | evgendvorkin & goodpjw2008 published "RSNA Versia_7"           |
| Pipeline Recipe   | Raptor-384 CoAtNet + Raptor-224 + Compact ConvNeXt R6 + Reader |
| The Uniform Trap  | 800+ teams ran the exact public blend with uniform weights      |
| Dilution Danger   | Adding 16-window non-OAI models at 15-50% dropped LB to 0.953  |
| The Winning Delta | Continuous Probit tail preservation + 12x12 Orthopaedic Matrix |
+-------------------+----------------------------------------------------------------+
```

### Why Gold-58 Simplex Optimization Was a Statistical Mirage
* **The Supervision Bottleneck**: The competition provided ~4,400 multi-series studies, but ONLY 58 studies officially carried radiologist-adjudicated labels ("Gold-58").
* **Small Sample Pathology Scarcity**:
  - `Bone Fracture`: ~3% prevalence $\to$ only **1 to 2 positive patients** in Gold-58.
  - `Baker's Cyst`: ~4% prevalence $\to$ only **2 positive patients** in Gold-58.
* **The Overfitting Trap**: Running continuous simplex optimization (e.g. Nelder-Mead on 24 model weights) on 58 samples overfits to the idiosyncratic features of 2 individual patients, leading to negative generalization on the ~400 Private Leaderboard studies.
* **The Correct Strategy**: Use Gold-58 strictly as an offline verification gate with paired bootstrap confidence intervals ($B=2,000$ iterations), while locking blend weights to domain-grounded specialist priors.

---

## Key Benchmarks & Leaderboard Progression

| Candidate Version | Key Architecture | Public LB | Private LB | Gold-58 AUC | Kaggle Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **V17 (Verified PB)** | Multi-Scale Raptor (384+224) + Heliosli R6 | **0.9580** | — | 0.9520 | 443.4 s |
| **V20 (Power Rank)** | Generalized Power-Mean ($p=1.40$) on Focal | **0.9580** | **0.9420** | 0.9530 | 469.4 s |
| **V45 (Sovereign)** | Dual-Arm Anchor + Biomechanical Coupling | **0.9580** | — | 0.9550 | 533.9 s |
| **V47 (Dilution)** | Added 16-window auxiliary model at 25% | 0.9530 | — | 0.9480 | 520.0 s |
| **V50 (Fallback)** | Queried test IDs in training `all_vols.npy` | 0.4910 | — | 0.7200 | 279.8 s |
| **V56 (Reader)** | 2.5D Reader blended at $W=0.15$ | 0.9570 | — | 0.9540 | 551.7 s |
| **V58 (Thrashing)**| 12 threads on 4 vCPUs (GIL contention) | 0.9570 | — | 0.9540 | 660.9 s |
| **Candidate V60** | Pure-OAI + Versia_7 Reader + Power-Probit | **0.9585+** | — | **0.9582** | **452.1 s** |
