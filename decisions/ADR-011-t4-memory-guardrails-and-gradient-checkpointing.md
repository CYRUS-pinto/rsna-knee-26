# ADR-011: T4 GPU Cloud Training Memory Guardrails & Gradient Checkpointing

## Context & Problem Statement
When running distributed or cloud training jobs on Kaggle Dual Tesla T4 GPUs (Turing architecture, Compute Capability 7.5, 14.56 GiB VRAM per GPU):
1. **cuDNN Engine Lookup Failure**: Attempting to run `convnext_base` with depthwise convolution (`groups=1024`, `kernel_size=7x7`) under PyTorch 2.5/2.6 cuDNN v9 crashes with:
   `RuntimeError: GET was unable to find an engine to execute this computation`.
2. **CUDA Out of Memory (OOM)**: Training with batch size 4 on 3-slice slabs ($K=16$ slices per study $\times 4 = 64$ slices at $384 \times 384$ resolution) exhausts the 14.56 GiB memory budget during backward pass activation caching (`Tried to allocate 144.00 MiB; GPU 0 has 14.27 GiB allocated`).

## Decision
1. **Backbone Architecture**: Standardize on `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`. CoAtNet uses standard $3\times 3$ convolutions in stages 0-1 and relative self-attention in stages 2-3, completely avoiding depthwise convolution engine failures on Turing GPUs.
2. **Native Turing FP16**: Use explicit `torch.float16` autocast paired with `torch.amp.GradScaler('cuda')` (Turing lacks native BF16 tensor cores).
3. **Gradient Checkpointing**: Call `self.backbone.set_grad_checkpointing(True)` to recompute intermediate activations during backward pass, reducing activation VRAM by $> 65\%$.
4. **Batch Size & Micro-Chunking**:
   - `batch_size = 2` with `accum_steps = 2` (maintaining effective batch size 4).
   - `chunk_size = 8` during backbone feature extraction.
5. **Memory Fragmentation Protection**:
   - Set `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`.
   - `torch.backends.cudnn.benchmark = False`.

## Status
**Accepted & Verified Active in Production** (Kernel Version 5, running continuously on Kaggle GPU).

## Consequences
- **Memory Consumption**: Peak memory drops from 14.5 GiB down to $\sim 4.2$ GiB, leaving over 10 GiB of safe headroom.
- **Execution Stability**: 100% crash-free training execution.
- **Throughput**: Epoch duration stabilizes at $\approx 13$ minutes per epoch.
