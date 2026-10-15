# Engine 3: 3D Slice-Transformer MIL Trainer (Colab Pro A100)

## Overview
Engine 3 provides a genuine 3rd orthogonal vision paradigm to break the 0.956 ceiling and advance toward 0.967+.

## Architecture
- **Backbone:** CoAtNet-2 384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`) with TF32 acceleration and gradient checkpointing.
- **MIL Head:** `SliceTransformerHead`
  - 1D learnable slice positional embeddings along the Z-axis (depth).
  - 2-layer, 8-head Transformer Encoder (`d_model=768, dim_feedforward=1536`).
  - Pre-layer norm and post-pooling layer norm for numerical stability.
  - Per-finding cross-attention pooling for all 12 abnormalities.
- **Loss:** Asymmetric Loss (ASL) with $\gamma_- = 2.0, \gamma_+ = 0.0$, downweighting easy negative slices to focus on hard pathology boundaries.

## Numerical Stability Guardrails (Zero-Divergence)
- **BF16 Direct Backprop:** On A100 GPUs, `torch.bfloat16` bypasses `GradScaler` entirely. GradScaler was designed for FP16 and causes gradient explosion when used with BF16 dynamic ranges.
- **Learning Rates:**
  - Backbone: `2e-5` (OneCycleLR with 15% warmup)
  - Head: `2e-4` (preventing early saturation of linear classifier weights)
- **Gradient Clipping:** Hard clamp at `max_norm=1.0`.
- **Budget Control:** Standardized to 8 epochs (~1.9 hours, ~12.8 compute units total), preventing wasteful credit depletion.
