# Pillar 4: Loss Functions, Target Engineering & Numerical Optimization

## 1. Executive Summary: The Soft Target Reality
In RSNA Knee Abnormality Detection, only **58 studies** have gold binary annotations ($y \in \{0, 1\}$). The remaining **4,349 studies** possess soft probability targets derived from radiology reports via LLM consensus distillation (`data/SUPER_CONSENSUS_labels.parquet`), where $y \in [0.0, 1.0]$ represents the probabilistic likelihood of pathology.

Training a neural network on 98.7% continuous soft labels requires distinct mathematical discipline compared to standard binary classification.

---

## 2. Comparative Evaluation of Loss Formulations

| Loss Function | Mathematical Handling of Continuous $y \in [0, 1]$ | Gradient Behavior on Ambiguous Cases | Risk of Divergence | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **A. Standard BCEWithLogitsLoss** | **Native & Exact:** Smooth cross-entropy between continuous probabilities $y$ and predicted $\sigma(x)$. | Monotonic, non-vanishing gradients across all margins. | **Zero:** Mathematically stable, convex in logits. | 🏆 **Primary Standard:** Rock-solid convergence. |
| **B. Asymmetric Loss (ASL)** | **Approximate:** Downweights negative margins via $(1 - \sigma(x))^{\gamma_-}$. | Suppresses low-probability continuous labels ($y < 0.15$), causing gradient truncation. | **High:** If paired with high learning rates ($>5 \times 10^{-4}$) and BF16, gradients explode or collapse to AUC 0.50. | ⚠️ **Use with extreme caution:** Only for final fine-tuning at $\text{LR} \le 5 \times 10^{-5}$. |
| **C. Focal Loss** | Modulates all probabilities via $(1 - p_t)^\gamma$. | Over-penalizes soft label variance near $y \approx 0.5$. | **Moderate:** Unstable when soft label noise is present. | ❌ **Reject:** Destabilizes soft target convergence. |
| **D. Mean Squared Error (MSE)** | Treats probabilities as continuous regression targets. | Vanishing gradients when $\hat{y}$ approaches 0 or 1. | **Low**, but poor calibration on rare extreme positive events. | ❌ **Reject:** Inferior to log-likelihood for probability calibration. |

---

## 3. Root Cause Analysis: The Colab A100 Divergence

During earlier Colab experimentation, a training run diverged at Epoch 2 (AUC collapsing to 0.5000). The mathematical post-mortem revealed three compounding errors:
1. **Asymmetric Loss on Continuous Targets:** ASL was applied with hard probability clipping ($\text{clip} = 0.05$) and asymmetric margin power $\gamma_- = 2.0$ directly on soft labels with mean $\bar{y} \approx 0.18$. This induced severe non-linear gradient spikes.
2. **Aggressive Learning Rate:** $\text{LR} = 10^{-3}$ with OneCycleLR was too high for an uninitialized 3D Transformer head layered on top of a frozen/fine-tuned backbone.
3. **GradScaler with BF16:** PyTorch's `torch.cuda.amp.GradScaler` is engineered specifically for IEEE FP16 (which has only 5 exponent bits). BF16 has 8 exponent bits (identical dynamic range to FP32) and does not need dynamic loss scaling. Dynamic scaling in BF16 causes scale factor overflow and numerical saturation.

---

## 4. Mathematical Discipline & Guardrails for All Training Scripts

1. **Default Loss:** Use `nn.BCEWithLogitsLoss()`. It handles continuous soft targets $y \in [0, 1]$ natively with zero margin truncation:
   $$\mathcal{L} = -\sum_{c=1}^{12} \left[ y_c \log \sigma(x_c) + (1 - y_c) \log (1 - \sigma(x_c)) \right]$$
2. **Learning Rate & Schedule:**
   - Backbone: Peak LR $\le 1 \times 10^{-4}$ with 5% linear warmup and Cosine Annealing.
   - Attention-MIL Head: Peak LR $\le 3 \times 10^{-4}$.
3. **Precision:** Native `torch.bfloat16` backpropagation without `GradScaler`.
4. **Validation Metric Calculation:** Compute ROC AUC directly on **raw logits** $x$, never on $\sigma(x)$, to eliminate numerical underflow/overflow ties near 0.0 and 1.0.
5. **Circuit Breaker:** Automatic abort if batch loss exceeds $5.0$ or produces NaN.
