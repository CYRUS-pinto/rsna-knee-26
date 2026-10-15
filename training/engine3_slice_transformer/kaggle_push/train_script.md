# Standalone Kaggle Cloud Trainer (`train_script.py`)

## 1. Overview & Operational Role
`train_script.py` is a standalone, self-contained Python training script specifically engineered to execute autonomously on Kaggle cloud accelerators (Dual T4 GPUs or P100) via the Kaggle API.

It packages:
1. Dynamic input mount discovery.
2. In-memory memory-mapped volume loading (`ConcatVols` over `all_vols.npy` + `extra_vols.npy`).
3. Stratified Group Cross-Validation partitioning.
4. Native FP16 mixed-precision training with gradient scaling.
5. Best out-of-fold checkpoint persistence to `/kaggle/working`.

---

## 2. Hardware Engineering & T4 GPU Stability
Kaggle GPU instances provide dual NVIDIA T4 GPUs (Turing architecture, compute capability 7.5, 15.0 GB VRAM per GPU).

### 2.1 Critical Memory & Stability Guardrails
| Challenge | Root Cause | Solution Implemented |
| :--- | :--- | :--- |
| **CUDA OOM** | 64 high-resolution $384\times 384$ slice activations held during backpropagation. | 1. `batch_size = 2` with `accum_steps = 2` (effective batch size 4).<br>2. `backbone.set_grad_checkpointing(True)`.<br>3. `chunk_size = 8` during feature extraction.<br>4. `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`. |
| **cuDNN Engine Failure** | $7\times 7$ depthwise conv with 1,024 groups in `convnext` lacks FP16 engine on Turing GPUs. | Replaced with competition-proven `coatnet_rmlp_2_rw_384` hybrid architecture. |
| **bfloat16 Emulation Slowdown** | Turing GPUs lack native BF16 tensor cores. | Explicit `torch.float16` autocast paired with `torch.amp.GradScaler('cuda')`. |
| **Hardcoded Path Failures** | Kaggle mounts community datasets with prefixed author paths. | Recursive `_find(*names)` dynamic discovery across `/kaggle/input`. |

---

## 3. Dataset & Training Pipeline Details

### 3.1 3-Slice Slab Extraction
For study volume $V \in \mathbb{R}^{D \times H \times W}$ and sampled center slice $c$:
$$\text{Slab}_c = \text{Stack}\left[V[c-1], V[c], V[c+1]\right] \in \mathbb{R}^{3 \times H \times W}$$
- Standardized to $[0, 1]$ floating-point range.
- Bilinearly interpolated to native $384 \times 384$ input resolution.

### 3.2 Slice Sampling Strategy
- **Training**: Samples $K=16$ slices uniformly from the anatomically valid bounding region (identified via `all_masks.npy` and `extra_masks.npy`) with random temporal jitter.
- **Evaluation / Validation**: Evaluates $K=24$ slices evenly spaced across the entire valid volume for deterministic out-of-fold scoring.

### 3.3 Loss Function & Optimization
- **Loss**: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`) trained on continuous soft target probabilities $[0, 1]$ from `SUPER_CONSENSUS_labels.parquet`.
- **Optimizer**: AdamW with differential learning rates:
  - Backbone: $\text{LR} = 3 \times 10^{-5}$, Weight Decay = $0.01$.
  - Transformer Head: $\text{LR} = 2 \times 10^{-4}$.
- **Scheduler**: Cosine Annealing LR over 6 epochs down to $\text{LR}_{\min} = 10^{-6}$.

---

## 4. Execution & Monitoring Commands
To dispatch and check the training run autonomously via Kaggle CLI:
```bash
# Push script and start cloud execution
kaggle kernels push -p training/engine3_slice_transformer/kaggle_push/

# Check live execution status
kaggle kernels status cyruspinto/rsna-knee-multiplane-engine3-train

# Download live stdout and stderr logs
kaggle kernels output cyruspinto/rsna-knee-multiplane-engine3-train -p .kaggle_output
```

Upon completion, the best fold 0 model weights are saved at:
`/kaggle/working/engine3_multiplane_fold0_best.pth`
