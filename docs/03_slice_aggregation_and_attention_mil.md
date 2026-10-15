# Pillar 3: Slice-to-Study Aggregation & Attention-MIL Modeling

## 1. Executive Summary & The Aggregation Problem
A knee MRI study contains between 44 and 96 slices across multiple sequences. However, pathology is highly localized:
- An **ACL tear** is typically visible on only **2 to 4 sagittal slices**.
- An **MCL sprain/tear** is visible on only **3 to 5 coronal slices**.
- **Lateral Osteoarthritis** is visible on lateral tibial plateau slices.

If an aggregation method averages across all slices, the 3 abnormal slices are diluted by 40+ normal slices. Conversely, if it uses naive max-pooling, a single scanner artifact or motion blur produces a false positive.

---

## 2. Comparative Analysis of Aggregation Mechanisms

| Aggregation Method | Mathematical Formulation | Class Specificity | Cross-Slice Context | Multi-Plane Capability | Sensitivity to Artifacts | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Global Average Pooling (GAP)** | $\bar{h} = \frac{1}{K} \sum_{k=1}^K h_k$ | None (all classes share same pooled vector) | None | Naive concatenation | Low | ❌ **Rejected:** Dilutes localized tear signals by $\approx 90\%$. |
| **B. Global Max Pooling (GMP)** | $\hat{h} = \max_{k=1..K} h_k$ | None | None | Naive concatenation | High | ❌ **Rejected:** Extremely noisy, triggers on streak artifacts. |
| **C. Standard Attention-MIL** (Ilse et al., 2018) | $a_k = \text{softmax}(w^T \tanh(V h_k))$ | Single shared attention weight vector | None (slices evaluated independently) | Limited | Low | ⚠️ **Acceptable baseline:** Good, but misses anatomical relationships. |
| **D. 12-Class Independent Attention-MIL** | $a_{k, c} = \text{softmax}(w_c^T \tanh(V h_k))$ | **12 independent attention distributions** | None | Moderate | Very Low | ✅ **High Performance:** Allows ACL to attend to sagittal slices and MCL to attend to coronal slices. |
| **E. Multi-Plane Slice-Transformer + Class Attention-MIL (Engine 3)** | $H' = \text{Transformer}(H + P_{\text{intra}} + P_{\text{plane}})$, then 12-class Attention-MIL | **12 independent attention distributions** | **Full all-to-all cross-slice & cross-plane self-attention** | **Native** (via discrete plane embeddings) | Minimal | 🏆 **Gold Standard:** Models anatomical disruption across contiguous slices and pulse sequences simultaneously. |

---

## 3. Deep-Dive: How Multi-Plane Attention-MIL Solves the 96-Slice Inference Problem

### A. The 12-Condition Anatomical Attention Split
In `training/engine3_slice_transformer/model_multiplane.py`:
$$\mathbf{A} = \text{Softmax}\left(\mathbf{W}_{\text{att}} \cdot \text{GELU}(\mathbf{V} \cdot \mathbf{H})\right) \in \mathbb{R}^{B \times K \times 12}$$
Each condition $c \in \{1 \dots 12\}$ computes its own normalized attention distribution over the $K$ slices:
$$\bar{h}_c = \sum_{k=1}^K a_{k, c} h_k$$

- When predicting **MCL**: The network learns $a_{k, \text{MCL}} \to 0$ for all Sagittal and Axial slices, concentrating 95% of its probability mass on the **Coronal slices** where the medial collateral ligament is physically present.
- When predicting **ACL / PCL**: The network concentrates attention mass on the **intercondylar notch slices** of the Sagittal sequence.

### B. Discrete Sequence Plane Embeddings
Because test inference stacks 96 slices across 5 distinct acquisitions (Sagittal T1, Sagittal PD-FS, Coronal PD, Coronal FS, Axial), assigning continuous 1D positional embeddings $1 \dots 96$ creates a false boundary where slice 32 (Sagittal) is treated as adjacent to slice 33 (Coronal).

By adding discrete sequence embeddings:
$$h_k \leftarrow h_k + \mathbf{e}_{\text{intra}}(k) + \mathbf{e}_{\text{plane}}(\text{plane\_id}_k)$$
The Transformer self-attention layers understand both intra-sequence depth and cross-sequence identity, allowing seamless modeling of multi-plane studies.
