# Architecture Decision Records (ADRs)

This directory maintains the immutable decision log for the RSNA 2026 Knee Abnormality Detection challenge. Every architectural pivot, failed experiment, stability guardrail, and mathematical finding is documented here.

## Index of Decisions

| ADR | Title | Status | Impact / Core Takeaway |
| :--- | :--- | :---: | :--- |
| **[ADR-001](ADR-001-dual-engine-5050-rank-fusion.md)** | Dual-Engine Uniform Percentile Rank Fusion | **Accepted** | Minimum variance ensembling ($w_1=w_2=0.50$) achieves **0.956 PB**. |
| **[ADR-002](ADR-002-excising-heuristic-priors-and-overfitting.md)** | Excising Clinical Heuristic Priors | **Accepted** | Post-processing priors invert rankings on discordant pairs and overfit Gold-58 sample noise. |
| **[ADR-003](ADR-003-colab-a100-numerical-stability-and-bf16.md)** | A100 BF16 Training Stability & Loss Stabilization | **Accepted** | Resolves Epoch 2 divergence: eliminates GradScaler on BF16, clamps ASL soft targets, scores raw logits. |
| **[ADR-004](ADR-004-train-inference-slice-distribution-alignment.md)** | Slice Distribution Alignment (44 vs 96 Slices) | **Accepted** | Prevents train/test domain shift; establishes Attention-MIL + BCE baseline before sequential transformers. |
| **[ADR-005](ADR-005-kaggle-api-token-sanitization.md)** | Kaggle Credential Sanitization Policy | **Accepted** | Eliminates plaintext tokens from notebooks and version control using Colab Secrets & getpass. |
