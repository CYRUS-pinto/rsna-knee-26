# Pillar 2: Backbone Model Architecture & Orthogonal Diversity

## 1. Executive Summary & The Problem of Error Correlation
In competitive machine learning, ensembling multiple models succeeds only when individual models have **uncorrelated error residuals**. If Model A and Model B make the exact same errors on false negatives, blending them reduces zero variance.

Currently:
- **Engine 1:** Raptor CoAtNet-3 (Hybrid CNN + RelAttention, trained on OAI data).
- **Engine 2:** Heliosli ConvNeXt-Base R7 (Pure ConvNet, $7 \times 7$ depthwise convolutions).

To break past **0.956** toward **0.963+ (Gold tier)**, Engine 3 must be a **genuinely orthogonal architecture**.

---

## 2. Comparative Evaluation of Candidate Architectures for Engine 3

| Model Family | Inductive Bias | Spatial Attention Mechanism | Feature Resolution | Error Independence vs Engines 1 & 2 | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. CoAtNet-4 / MaxViT** | Hybrid Conv-Transformer | RelAttention / MBConv | $384 \times 384$ | **Low ($\rho \approx 0.94$)**: Redundant with Engine 1. | ❌ **Reject:** Duplicates Raptor's inductive bias. |
| **B. ConvNeXt-Large** | Pure ConvNet | $7 \times 7$ Depthwise Conv | $384 \times 384$ | **Moderate ($\rho \approx 0.88$)**: Shares inductive bias with Engine 2. | ⚠️ **Backup:** Strong single model, but limited ensemble lift. |
| **C. DINOv2-Base (ViT-B/14)** | Self-Supervised Vision Transformer | Global Self-Attention on $14 \times 14$ patches | $392 \times 392$ | **Very High ($\rho \approx 0.71$)**: Learns dense, unsupervised structural representations without task bias. | 🏆 **Top Choice:** Unlocks true orthogonal representation. |
| **D. Swin Transformer (Swin-Base)** | Hierarchical Shifted Windows | Local windowed attention | $384 \times 384$ | **High ($\rho \approx 0.76$)**: Multiscale hierarchical features capture both micro-tears and macroscopic joint effusion. | 🥈 **Strong Contender:** Excellent complementary visual feature hierarchy. |
| **E. DenseNet-121 / 201** | Dense Feature Reuse | Dense concatenation | $384 \times 384$ | **High ($\rho \approx 0.78$)**: Direct skip connections preserve sharp gradient edges for thin meniscus tears. | 🥉 **Viable Lightweight Alternative:** Low compute overhead. |

---

## 3. Why DINOv2 (Vision Transformer) & Swin-T Break the 0.956 Ceiling

### A. Non-Overlapping Patch Tokenization vs Convolutional Sliding Windows
- Both CoAtNet and ConvNeXt utilize convolutional inductive biases (local translation invariance). While excellent for general anatomy, they struggle with global multi-compartment interactions (e.g., how Lateral OA correlates with Medial Meniscus extrusion).
- DINOv2 decomposes slices into $14 \times 14$ non-overlapping spatial patches and executes all-to-all global self-attention across the entire joint space.
- Features extracted from DINOv2 are completely self-supervised on 142M curated natural and scientific images, making them robust to MRI scanner noise, artifact streaks, and foreign field inhomogeneities.

### B. Computational Budget & Training on Kaggle GPUs
- DINOv2-Base has 86M parameters (smaller than ConvNeXt-Large's 198M).
- Using pre-extracted features or fine-tuning with LoRA / frozen backbone heads allows 5-fold training in **under 2 hours on a single Kaggle T4 GPU**.

---

## 4. Implementation Strategy for the Team
- **Work Stream 1:** Teammate 2 fine-tunes DINOv2-Base on the 5-fold partitioned dataset (`data/train_folds.parquet`).
- **Work Stream 2:** Teammate 3 fine-tunes Swin-Base as an alternate candidate.
- **Integration:** The best performing model on the 5-fold CV is integrated into Engine 3 of the Tri-Hybrid ensemble.
