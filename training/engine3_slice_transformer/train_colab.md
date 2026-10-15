# Colab Pro / Lambda Trainer (`train_colab.py`)

## 1. Overview
`train_colab.py` is an optimized training script designed for high-memory cloud accelerators (such as Google Colab Pro A100 80GB, L4, or Lambda Cloud nodes).

It supports:
- Bfloat16 direct backprop (when running on Ampere / Ada / Hopper architectures).
- Asymmetric Loss (ASL) for hard-negative margin optimization.
- Automatic corpus retrieval directly from Kaggle Datasets via the Kaggle API.
- Stochastic Weight Averaging (SWA) over the final 3 epochs.

---

## 2. Supported Loss Functions

### 2.1 Standard BCEWithLogitsLoss
Standard binary cross-entropy with optional positive class weighting:
$$\mathcal{L}_{\text{BCE}} = -\sum_{j=1}^{12} \left[ w_j y_j \log \sigma(s_j) + (1 - y_j) \log (1 - \sigma(s_j)) \right]$$
where $w_j = \text{clip}\left(\frac{1 - \text{prev}_j}{\text{prev}_j}, 1.0, 10.0\right)$ counteracts extreme class imbalance.

### 2.2 Asymmetric Loss (ASL)
Formulated by Ridnik et al. (ICCV 2021) for multi-label classification where negative samples vastly outnumber positive samples:
- Dynamically downweights easy negative samples: $\gamma_{\text{neg}} = 2.0, \gamma_{\text{pos}} = 0.0$.
- Uses probability margin clipping ($m = 0.05$) to zero out gradients from extremely confident easy negatives.

---

## 3. Stochastic Weight Averaging (SWA)
After completing training across all epochs, the model calculates the arithmetic average of model weights across the final 3 stable checkpoints:
$$\mathbf{W}_{\text{SWA}} = \frac{1}{3} \sum_{e = E-2}^E \mathbf{W}_e$$
SWA centers the weights in a flatter minimum in the loss landscape, yielding $+0.001 - 0.003$ gains in out-of-fold generalization without increasing inference latency.
