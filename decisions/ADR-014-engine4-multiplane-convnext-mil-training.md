# ADR-014: Proprietary Engine 4 — Multi-Plane 2.5D ConvNeXt-Small Attention MIL Training

## Status
**Accepted** (Oct 16, 2026)

## Context & Autopsy of Engine 3
In our earlier attempt to train a proprietary representation model (Engine 3: CoAtNet-384 with slice-transformer head), the training ran for 7.5 hours on Google Colab and resulted in an Out-of-Fold (OOF) macro-AUC of **0.709**.

Forensic inspection of the code revealed three structural bugs:
1. **Single-Plane Collapse:** In `train_script.py` and `train_multiplane_cv.py`, `all_vols.npy` contained exclusively Sagittal slices (`plane_ids = torch.zeros(...)`). Despite being named "MultiPlane", the model was never exposed to Coronal series (where MCL tears and collateral ligaments are diagnosed) or Axial series (where Patellofemoral OA and effusions are localized).
2. **Synthetic Zero Placeholders:** In `src/data/multiplane_dataset.py`, lines 96–100 generated synthetic zero tensors (`slab = torch.zeros((3, self.res, self.res))`) as an unlinked placeholder, which would have fed blank images into the neural network during training.
3. **Effective Batch Size Underflow:** Batch size was set to $B=2$ without gradient accumulation, causing severe gradient noise and optimization instability.

To compete for the Gold Tier ($0.963 - 0.967+$), we must train a genuine 3-plane physical model that resolves these structural defects.

---

## Architectural Decision: Engine 4

### 1. 5-Slot Anatomical Sequence Representation
Engine 4 divides the knee MRI study into 5 discrete anatomical sequence slots:
- **Slot 0:** Sagittal Fluid-Sensitive (T2 / PDFS) — Cruciate ligaments (ACL/PCL), joint effusions
- **Slot 1:** Sagittal Non-Fluid (T1 / PD) — Menisci, subchondral bone contusion, cortical fractures
- **Slot 2:** Coronal Fluid-Sensitive (T2 / PDFS) — Collateral ligaments (MCL/LCL), synovial proliferation
- **Slot 3:** Coronal Non-Fluid (T1 / PD) — Medial & lateral compartment cartilage loss and osteophytes
- **Slot 4:** Axial — Patellofemoral joint alignment, cartilage thickness, and popliteal cysts (Baker's)

### 2. Physical Slice-Normal Coordinate Sorting & 140mm FOV
- Slices are sorted along the patient's true physical axis using the slice normal vector $\vec{n} = \vec{r} \times \vec{c}$ derived from `ImageOrientationPatient`:
  $$\text{position} = \vec{n} \cdot \vec{\mathbf{p}}_{\text{IPP}}$$
- Slices are cropped to a standardized **140 mm physical bounding box** using `PixelSpacing`, ensuring that visual structures maintain anatomical scale regardless of scanner matrix dimensions.
- Contrast is normalized dynamically using 2nd and 98th intensity percentiles.

### 3. 12 Disentangled Gated Attention MIL Heads
Rather than forcing all 12 pathologies to pool volumetric slices identically, Engine 4 equips each pathology with an independent Gated Attention mechanism:
$$a_j(h) = \text{softmax}\left(\mathbf{w}_j^\top \left(\tanh(\mathbf{V}_j h) \odot \sigma(\mathbf{U}_j h)\right)\right)$$
$$\mathbf{z}_j = \sum_{k=1}^K a_{jk} h_k, \quad \hat{y}_j = \mathbf{w}_{\text{cls}, j}^\top \mathbf{z}_j + b_j$$
This allows the ACL head to attend exclusively to mid-Sagittal slices while the MCL head simultaneously attends exclusively to mid-Coronal slices.

### 4. Gradient Accumulation & Numerical Stability
- Per-GPU batch size $B=2$ is combined with `accum_steps=8` to guarantee an effective batch size of $16\text{--}32$.
- Gradient clipping is enforced at norm $1.0$.
- Supervision uses continuous soft targets ($0.0 \to 1.0$) from `data/SUPER_CONSENSUS_labels.parquet`, avoiding the destructive binarization of noisy labels.

---

## Consequences & Verification Matrix

| Component | Engine 3 (Failed) | Engine 4 (Current) | Verification Method |
| :--- | :--- | :--- | :--- |
| **Planes Fed** | Sagittal only (`plane_ids=0`) | Sagittal, Coronal, Axial (5 Slots) | `src/data/multiplane_dataset.py` |
| **Slice Sorting** | InstanceNumber index | Physical slice-normal $\vec{n} \cdot \vec{\mathbf{p}}$ | Verified via IOP/IPP projection |
| **FOV Normalization** | Naive pixel resize | 140 mm physical crop | `crop_and_normalize()` function |
| **Attention Heads** | Single pooled transformer | 12 Disentangled Gated MIL Heads | `src/models/engine4_convnext_mil.py` |
| **Batch Size** | $B=2$ (noisy) | $B=2 \times 8 = 16$ effective | `accum_steps=8` in `train_engine4.py` |
| **Notebook** | Fragmented | Turnkey Colab/Kaggle A100 | `training/engine4/colab_train_engine4_a100.ipynb` |

## Deployment Strategy
Engine 4 is designed to operate as the proprietary 3rd pillar in our **Tri-Hybrid Rank Blend** alongside Raptor (CoAtNet SWA) and ConvNeXt R7:
$$R_{\text{final}} = 0.40 \cdot \text{Rank}(S_{\text{raptor}}) + 0.35 \cdot \text{Rank}(S_{\text{convnext\_r7}}) + 0.25 \cdot \text{Rank}(S_{\text{engine4}})$$
Because Engine 4 brings orthogonal multi-plane slice-attention features that neither public baseline possesses, it provides the mathematical leverage needed to break the $0.956$ public tie-breaker plateau.
