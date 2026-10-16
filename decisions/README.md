# Architecture Decision Records (ADRs)

This directory maintains the immutable decision log for the RSNA 2026 Knee Abnormality Detection challenge. Every architectural pivot, failed experiment, stability guardrail, and mathematical finding is documented here.

## Index of Decisions

| ADR | Title | Status | Impact / Core Takeaway |
| :--- | :--- | :---: | :--- |
| **[ADR-001](ADR-001-dual-engine-5050-rank-fusion.md)** | Dual-Engine Uniform Percentile Rank Fusion | **Accepted** | Minimum variance ensembling ($w_1=w_2=0.50$) achieves **0.956 PB**. |
| **[ADR-002](ADR-002-excising-heuristic-priors-and-overfitting.md)** | Excising Clinical Heuristic Priors | **Accepted** | Empirical proof (V12 dropped to 0.953): priors invert rankings and overfit Gold-58 sample noise. |
| **[ADR-003](ADR-003-colab-a100-numerical-stability-and-bf16.md)** | A100 BF16 Training Stability & Loss Stabilization | **Accepted** | Direct BF16 backprop, clamped soft targets, and raw logit evaluation eliminate training divergence. |
| **[ADR-004](ADR-004-train-inference-slice-distribution-alignment.md)** | Slice Distribution Alignment (44 vs 96 Slices) | **Accepted** | Prevents train/test domain shift; establishes Attention-MIL + BCE baseline before sequential transformers. |
| **[ADR-005](ADR-005-kaggle-api-token-sanitization.md)** | Kaggle Credential Sanitization Policy | **Accepted** | Eliminates plaintext tokens from notebooks and version control using Colab Secrets & getpass. |
| **[ADR-006](ADR-006-input-representation-and-spatial-geometry.md)** | Input Representation & 3D Spatial Geometry | **Accepted** | 2.5D multi-plane slabs with 3D normal vector slice sorting saves +0.028 Macro-AUC over random indexing. |
| **[ADR-007](ADR-007-orthogonal-backbone-selection.md)** | Orthogonal Backbone Triad (CoAtNet + Swin + DINOv2) | **Accepted** | Eliminates redundant ConvNets; pairs global hybrid transformers with dense local feature extractors. |
| **[ADR-008](ADR-008-cross-plane-attention-mil-pooling.md)** | Cross-Plane Attention-MIL with Plane Embeddings | **Accepted** | Sequence-aware Transformer Head models contiguous anatomical disruption across all 3 planes. |
| **[ADR-009](ADR-009-continuous-soft-target-loss-discipline.md)** | Continuous Soft-Target Supervision & Logit Scoring | **Accepted** | BCE on continuous targets preserves uncertainty; raw logit evaluation prevents sigmoid saturation ties. |
| **[ADR-010](ADR-010-efficiency-prize-and-rank-fusion.md)** | $18,000 Efficiency Prize Compliance & Uniform 1/3 Fusion | **Accepted** | Uniform percentile rank ensembling guarantees monotonic ordering while staying under $T \le 450$s limit. |
| **[ADR-011](ADR-011-t4-memory-guardrails-and-gradient-checkpointing.md)** | T4 GPU Cloud Memory Guardrails & Gradient Checkpointing | **Accepted** | Gradient checkpointing, native FP16 GradScaler, and chunk_size=8 eliminate cuDNN crashes and CUDA OOM. |
| **[ADR-012](ADR-012-interactive-multiplane-attention-architecture-visualizer.md)** | Interactive Multi-Plane Visualizer Architecture | **Accepted** | Standalone browser-executable HTML visualizer (`docs/multiplane_transformer_visualizer.html`) for 12 findings. |
| **[ADR-013](ADR-013-grandmaster-blueprint-pseudo-labeling-and-anatomical-rectification.md)** | Grandmaster Blueprint: Pseudo-Labeling, Rectification & Specialist Routing | **Accepted** | Iterative Bayesian pseudo-labeling, laterality rectification, dual-stream FOV, SWA, and 4-focal/8-systemic column routing. |
| **[ADR-014](ADR-014-engine4-multiplane-convnext-mil-training.md)** | Proprietary Engine 4: Multi-Plane 2.5D ConvNeXt Attention MIL | **Accepted** | True 5-slot multi-plane physical sorting, 12 disentangled Gated Attention MIL heads, and continuous soft supervision to break the 0.956 plateau. |
| **[ADR-015](ADR-015-plane-decoupled-specialists-anti-collapse.md)** | Plane-Decoupled Finding Specialists & Anti-Collapse Protocol | **Accepted** | Decouples Sagittal (cruciate/menisci/contusion), Coronal (collateral/OA), and Axial specialists; lifts Sagittal ACL to 0.8636. |
| **[ADR-016](ADR-016-strategic-triage-and-v17-deployment-truth.md)** | Strategic Triage, Engine 5 Post-Mortem & V17 Inference Truth | **Accepted** | Forensic audit of Engine 5 dataset slice variance & missing test DICOM ETL; confirms V17 443.4s dual-arm pipeline and locks competition strategy. |


