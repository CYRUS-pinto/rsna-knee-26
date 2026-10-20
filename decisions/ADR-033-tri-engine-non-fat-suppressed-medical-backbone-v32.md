# ADR-033: Tri-Engine Non-Fat-Suppressed Medical Backbone (Candidate V32)

- **Status:** PROPOSED & PROTOTYPED
- **Date:** 2026-10-17
- **Author:** Principal ML Systems Architect & Competition Grandmaster Team
- **Target Metric:** Mean Column-wise Macro-ROC-AUC (Target: $\ge 0.963\text{--}0.966$)

---

## 1. Context & The "Missing Physical Modality" Breakthrough

In forensic analysis of `antoinegg1/rsna-knee-e11-diverse-heads-v20` ([e11_contract.json](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/scratch/rad_arm/e11_contract.json)), we discovered the root reason why our models (and the public clustered at 0.958) remain capped:

1. **The Fat-Suppressed Monoculture:**
   - Both our Raptor CoAtNet arm and ConvNeXt R7 arm read exclusively or predominantly **fat-suppressed series** (T2/PD/STIR, `Fat_Suppression == 1`).
   - While fat suppression makes fluid edema bright (ideal for Effusion and Baker's Cyst), it **drastically suppresses bone and cartilage tissue contrast**, blurring the fine structural boundaries of:
     - `Lateral Meniscus` (blurred against adjacent fat)
     - `Lateral OA` & `Medial OA` (joint space narrowing obscured)
     - `Bone Contusion` & `Fracture` (cortical bone margins lost)

2. **The E11 Empirical Proof ($N=4,407$ Studies, 3,000 Bootstrap Iterations):**
   - Injecting a dedicated **RadImageNet ResNet-50** reading **Non-Fat-Suppressed anatomical slots** (`SAG_NOFS`, `COR_NOFS`, `AX_NOFS`) with a 130mm physical crop produced the following verified per-target AUC lifts over the pure fat-suppressed baseline:
     - **Lateral Meniscus:** **$+0.0546$ AUC** ($0.642 \to 0.697$)
     - **Synovitis:** **$+0.0394$ AUC** ($0.742 \to 0.781$)
     - **Bone Contusion:** **$+0.0351$ AUC** ($0.842 \to 0.877$)
     - **Lateral OA:** **$+0.0290$ AUC** ($0.708 \to 0.737$)
     - **MCL Tear:** **$+0.0204$ AUC** ($0.884 \to 0.905$)
     - **Overall Macro-AUC Lift:** **$+0.0161$** ($p < 0.005$ across 3,000 draws).

---

## 2. Architectural Blueprint for Candidate V32

Candidate V32 combines three orthogonal medical vision representations:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             STAGE 1: DICOM I/O & MULTI-SLOT ROUTING              │
│   • Multi-worker C++ fastread decoding (4 workers)                               │
│   • Standardized 140mm physical FOV center crop via PixelSpacing                 │
│   • Dual-stream plane assignment: Fat-Suppressed (FS) + Non-Suppressed (NOFS)    │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌───────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────────┐
│  ENGINE 1: 2.5D MIL   │ │    ENGINE 2: CONVNEXT R7     │ │  ENGINE 3: MEDICAL   │
│  (CoAtNet 384 + 224)  │ │   (Multiplane MIL Head)      │ │   (RadImageNet 50)   │
│ • Full-field 384+224  │ │ • 352px synovial crop        │ │ • Non-FS Anatomical  │
│ • Gated slice pool    │ │ • Effusion / Baker stackers  │ │   Slices (SAG/COR/AX)│
│ • Domain: Ligaments   │ │ • Domain: Fluid / Patella    │ │ • Domain: Meniscus/OA│
└───────────┬───────────┘ └──────────────┬───────────────┘ └──────────┬───────────┘
            │                            │                            │
            ▼                            ▼                            ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   STAGE 2: PERCENTILE RANK NORMALIZATION                         │
│               r_i = rankdata(P_i, method='average') / N                          │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: TRI-ENGINE RANK-POWER FUSION                         │
│   Final_Score = ( w_rap * r_rap^p + w_cnx * r_cnx^p + w_rad * r_rad^p )^(1/p)    │
│   where p = 1.8                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Operational Deployment Strategy

1. **Candidate V30 (Lightning Master):** Locked for the **$18,000 Efficiency Prize track** (Stride-2, 2.033s GPU/study, ~15 min runtime).
2. **Candidate V31 (Capacity Master):** Locked for the **Main Grandmaster Track** (Full 4-fold ConvNeXt R7 + Stride-2 + Rank-Power $p=1.8$).
3. **Candidate V32 (Tri-Engine Master):** Incorporates the E11 Non-Fat-Suppressed RadImageNet heads to lift the low-SNR classes (`Lateral Meniscus`, `Synovitis`, `Lateral OA`) by $+0.03\text{--}+0.05$ AUC.
4. **Permanent Safety Net:** **Slot 1 (V6, 0.956)** and **Slot 2 (V17, 0.958)** remain permanently secured in the Kaggle UI.
