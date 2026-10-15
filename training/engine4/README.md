# Engine 4: Multi-Plane 2.5D ConvNeXt-Small Attention MIL

## 1. Overview
Engine 4 is our proprietary deep representation model designed to break the public 0.956 tie-breaker plateau by solving the three core failure modes of previous pipelines:
1. **Multi-Plane Coverage:** Ingests slices across 5 distinct sequence slots (Sagittal T1/T2, Coronal T1/T2, Axial) rather than relying on Sagittal-only data.
2. **Disentangled Per-Finding Attention:** Equips each of the 12 target pathologies with its own independent Gated Attention mechanism ($V$ and $U$ branches).
3. **Continuous Soft Target Supervision:** Trained on `data/SUPER_CONSENSUS_labels.parquet` (4,407 continuous multi-reader soft probabilities), completely avoiding noisy label binarization.

---

## 2. File Manifest

| File | Purpose |
| :--- | :--- |
| **[`train_engine4.py`](train_engine4.py)** | Main PyTorch training harness with gradient accumulation (`accum_steps=8`), AMP mixed precision, StratifiedGroupKFold on `StudyInstanceUID`, and exact Wilcoxon-Mann-Whitney AUC logging. |
| **[`colab_train_engine4_a100.ipynb`](colab_train_engine4_a100.ipynb)** | Turnkey Google Colab Pro (A100/L4) / Kaggle GPU execution notebook. |
| **[`../../src/data/multiplane_dataset.py`](../../src/data/multiplane_dataset.py)** | Physical 2.5D slice triplet extractor with slice normal sorting along $\vec{n} = \vec{r} \times \vec{c}$, 140mm FOV crop, and 2nd/98th percentile contrast normalization. |
| **[`../../src/models/engine4_convnext_mil.py`](../../src/models/engine4_convnext_mil.py)** | ConvNeXt-Small backbone with gradient checkpointing, 5-slot learnable plane embeddings, and 12 gated attention pooling heads. |
| **[`../../decisions/ADR-014-engine4-multiplane-convnext-mil-training.md`](../../decisions/ADR-014-engine4-multiplane-convnext-mil-training.md)** | Full architectural decision record and failure autopsy of Engine 3. |

---

## 3. How to Launch Training

### Option A: Google Colab Pro (A100 / L4 GPU)
Open [`colab_train_engine4_a100.ipynb`](colab_train_engine4_a100.ipynb) in Colab and run all cells sequentially.

### Option B: Command Line (Kaggle Dual T4 or Colab Terminal)
```bash
python training/engine4/train_engine4.py \
    --fold 0 \
    --epochs 6 \
    --batch_size 2 \
    --accum_steps 8 \
    --lr 1e-4 \
    --data_path data/ \
    --dicom_dir /kaggle/input/rsna-knee-abnormality-detection/train_images
```

---

## 4. Integration into Slot 2 Master Blend

Once Fold 0 training completes and generates `weights/engine4_convnext_fold0_best.pth` and `oof/oof_engine4_fold0.parquet`, it integrates into the **Tri-Hybrid Rank Blend**:
$$R_{\text{final}} = 0.40 \cdot \text{Rank}(S_{\text{raptor}}) + 0.35 \cdot \text{Rank}(S_{\text{convnext\_r7}}) + 0.25 \cdot \text{Rank}(S_{\text{engine4}})$$

Because Engine 4 provides independent 3-plane attention representations that neither public baseline possesses, the resulting ensemble breaks through the 0.956 tie-breaker ceiling while maintaining runtime well under the 500s threshold ($T \approx 420$s).
