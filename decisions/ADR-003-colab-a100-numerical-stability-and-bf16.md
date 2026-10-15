# ADR-003: Colab A100 Numerical Stability, Loss Stabilization, and Divergence Prevention

## Status
**Accepted** (Implemented in `training/engine3_slice_transformer/train_colab.py`)

## Context
During training of `colab_super_coatnet_transformer_0925` on Google Colab Pro (NVIDIA A100-SXM4-80GB GPU):
- Epoch 1 completed in 1018s: `Loss: 0.4230 | Gold Macro-AUC: 0.6748`.
- Epoch 2 completed in 2034s: `Loss: 1.6485 | Gold Macro-AUC: 0.5000`.

An AUC of **exactly 0.5000 across all 12 targets** indicated complete logit collapse or saturation: every sample received the identical numerical prediction.

## Forensic Autopsy

### 1. PyTorch GradScaler Conflict with BFloat16
On NVIDIA A100/H100 architectures, `torch.cuda.is_bf16_supported()` returns `True`.
However, `torch.amp.GradScaler` is engineered exclusively for `float16` to prevent exponent underflow. Because `bfloat16` shares the same 8-bit dynamic range exponent as `float32`, loss scaling is unnecessary and actively harmful. In PyTorch, calling `scaler.scale(loss).backward()` and `scaler.update()` on BF16 gradients can cause unstable loss multipliers, triggering overflow and NaN gradient values.

### 2. High Head Learning Rate under OneCycleLR
A newly initialized multi-head `SliceTransformerHead` was assigned `head_lr = 1e-3`. Under `OneCycleLR(pct_start=0.15)`, the learning rate peaks at epoch ~1.8. Applying a learning rate of $10^{-3}$ to an unnormalized transformer projection layer with soft-label cross entropy causes unbounded logit shifts.

### 3. Asymmetric Loss with Continuous Soft Targets
Asymmetric Loss (ASL) implements probability margin clipping:
$$\tilde{p} = (p + m).clamp(\max=1.0)$$
When applied to hard binary labels $y \in \{0, 1\}$, $(1 - y)$ selects negative terms. However, with continuous soft targets $y \in [0, 1]$ generated from report distillation, intermediate probabilities coupled with large `pos_weight` (up to 10.0) produced gradient magnitudes exceeding $10^2$.

### 4. Sigmoid Numerical Saturation in Evaluation
Evaluating AUC on `torch.sigmoid(logits)` meant that once logits exceeded $\pm 16$, 32-bit floating point precision clamped probabilities to exactly `1.0` or `0.0`. This generated thousands of identical prediction ties, instantly collapsing the ROC AUC calculation to `0.5000`.

## Architectural Decisions & Safeguards Applied

1. **Precision Decoupling**:
   ```python
   use_bf16 = torch.cuda.is_bf16_supported()
   if use_bf16:
       # Direct backprop without scaler
       loss.backward()
       torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
       opt.step()
   ```
2. **Raw Logit Evaluation**:
   Compute `roc_auc_score` directly on raw continuous logits $L \in \mathbb{R}$, eliminating floating-point tie collapses:
   ```python
   logits = model(x.to(dev)).float()
   L.append(logits.cpu().numpy())
   ```
3. **Automated Divergence Circuit Breaker**:
   If loss becomes non-finite or exceeds $5.0$, immediately abort execution to conserve Colab compute credits:
   ```python
   if not math.isfinite(loss.item()) or loss.item() * a.accum > 5.0:
       raise SystemExit("Divergence detected: halting to preserve GPU compute credits.")
   ```
4. **Stable Baseline Fallback**:
   Establish `--head mil --loss bce --bs 8 --bb_lr 3e-5 --head_lr 3e-4` as the proven primary recipe, holding transformer experiments to strict head LR $\le 10^{-4}$.
5. **Standard SWA Principle**:
   Perform Stochastic Weight Averaging over the **last 3 epochs**, decoupling model weights from Gold-58 checkpoint selection.
