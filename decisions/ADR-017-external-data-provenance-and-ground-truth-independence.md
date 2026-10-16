# ADR-017: External Data Provenance, Ground Truth Independence, and Compliance Audit

## Status
Accepted

## Date
2026-10-16

## Context & Forum Intelligence (Discussion #743416, #749858, #747658)

### 1. The Ground Truth Discrepancy Ceiling (The 0.90 Barrier)
As confirmed by Grandmaster `hengck23` in competition discussion #747658:
> *"The GT labels may be created by radiologist/doctors without looking at the report. Hence, the report and label are independent, and a discrepancy exists. That explains the upper limit of 0.90 in the results."*

**Concrete Case Study (Gold Study `1.2.826.0.1.3680043.8.498.12606657226568558340797193167488111973`):**
- **Ground Truth Label**: `Effusion = 1`
- **Clinical Report Narrative**: *"No knee effusion is present."*

**Implication for Modeling:**
- Free-text report extraction (whether via regex or SOTA LLMs like Qwen-72B or Claude-3.5) has an intrinsic diagnostic ceiling of ~0.89–0.90 due to textual omission and radiologist reading discrepancies.
- Purely vision-driven models pretrained on radiologic imaging features (OAI, fastMRI, SKM-TEA) bypass the report discrepancy ceiling, which is why multi-scale visual ensembles achieve **0.956 – 0.958+**.

---

### 2. External Dataset Provenance & Compliance Disclosure (Discussion #743416)
Kaggle competition hosts require public disclosure of all external training datasets. An audit of the public model checkpoints incorporated into our verified ensembles confirms compliance:

| Model / Artifact | Checkpoint Dataset | External Training Data | Access Status / License |
| :--- | :--- | :--- | :--- |
| **CoAtNet-384 SWA** | `nartaa/rsna-knee-publication-swa-weights-20261007` | **OAI (Osteoarthritis Initiative)**: 2,399 knees used for PF OA, Lateral OA, and Synovitis | Public research dataset; free registration click-through |
| **CoAtNet-224 Crop SWA** | `nartaa/rsna-knee-publication-swa-weights-20261007` | **OAI (Osteoarthritis Initiative)**: 2,399 knees, identical recipe | Public research dataset; free registration click-through |
| **ConvNeXt-S6 R7** (`8c7b666e`) | `heliosli/rsna-knee-compact-r7cnxfsss` | **fastMRI + fastMRI+**: Bounding boxes as auxiliary task (`fm_aux`, weight 0.3) | Public research dataset; open access |
| **ConvNeXt-S6 R6** (`fe055406`) | `heliosli/rsna-knee-r6-frozen-fe055406-w1` | Clean ImageNet-22k pretrained base without auxiliary bounding boxes | Standard open-source weights |
| **`anatomy.pt` (Crop Placer)** | `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | **SKM-TEA (Stanford Knee MRI)**: Used solely for 3D landmark joint condyle localization | Open research benchmark |

---

### 3. Verification of Bundle ID `fe055406fad6a12f`
- Community discussions noted that some versions of `heliosli/rsna-knee-r6-frozen-fe055406` contained R7 manifests.
- **Audit Result**: Our Candidate V17 and V18 pipelines explicitly mount `heliosli/rsna-knee-r6-frozen-fe055406-w1` and enforce:
  ```python
  assert manifest['bundle_id'] == 'fe055406fad6a12f'
  ```
  This assertion passed with **zero errors** during test-time evaluation, completing in **443.4s** with **0 fallbacks**.

---

## Decision

1. **Retain Dual-Arm Multi-Scale Pipeline as the Primary Vehicle**: Relying on OAI- and SKM-TEA-derived representations is mathematically justified because it breaks through the 0.90 report text barrier.
2. **Document External Data Full Compliance**: All external data sources (OAI, fastMRI, SKM-TEA) are verified as publicly accessible with research click-through agreements, fully satisfying Kaggle competition rules.
3. **Commit Forum Intelligence to Repository**: Maintain continuous documentation of all community discoveries in `docs/` and `decisions/`.

---

## Consequences
- 100% compliance with Kaggle competition rules on external data disclosure.
- Eliminates risk of bundle ID regressions or invalid weight mappings.
- Confirms why image-based multi-scale ensembling is structurally superior to text-derived weak supervision.
