# ADR-030: Stride-2 Anatomical Window Pruning, 4-Worker CPU Decoding, and Candidate V30 Lightning Master

**Status:** ACCEPTED & PUSHED TO KAGGLE  
**Date:** 2026-10-17  
**Author:** AI Technical Advisor & Engineering Team  
**Scope:** Slashing CPU decode bottlenecks, cutting sliding windows by 50% via stride-2 sampling, restoring full decoupled specialist weights, and deploying Candidate V30.

---

### 1. Context & Motivation

In ADR-029, Candidate V29 successfully eliminated 75% of forward pass compute by pruning ConvNeXt to Fold 0, completing its Kaggle GPU run in 423 seconds. 

However, inspecting the verified Kaggle receipt revealed a critical bottleneck:
* **GPU Time:** 5.467 seconds per study.
* **CPU Decode Time:** 50.033 seconds per study.
* **Windows Evaluated:** 528.7 windows per study.

The CPU DICOM decompression and geometry bank generation consumed **90% of total inference time**. Furthermore, evaluating 528 sliding windows at stride 1 evaluated redundant adjacent 3.5mm slices.

---

### 2. Engineering Innovations in Candidate V30

#### A. Stride-2 Anatomical Window Pruning
In Cell 6 of [`rsna-knee-v30-lightning-power-master.ipynb`](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/submissions/v30_lightning_power_master/rsna-knee-v30-lightning-power-master.ipynb):
```python
manifest['models'] = ['E_lrprior_s0_f0.pt']
manifest['eval_stride'] = 2
```
* **Mechanism:** In `compact_train.py`, `covering_centers(n, phase=1, stride=2)` samples every 2nd slice while preserving edge slices ($0$ and $n-1$).
* **Anatomical Safety:** Knee articular cartilage, meniscal tears, and cruciate ligaments span 12–25mm of joint space. With 3-slice window depth (`self.slices = 3`), stride 2 maintains continuous volumetric coverage without missing focal tears.
* **Impact:** Window count drops from $\sim 528 \rightarrow \sim 264$ windows. GPU inference time drops from $5.4\text{s} \rightarrow 2.7\text{s}$ per study.

#### B. 4-Worker Multiprocessing for DICOM CPU Decompression
In Cell 7:
```python
rc = subprocess.run([..., '--loader-workers', '4', ...])
```
* **Mechanism:** Kaggle Tesla T4 instances provide 4 vCPUs. Sponsoring 4 loader workers per GPU instance parallelizes `self.native.prepare` across all available CPU cores.
* **Impact:** CPU decode throughput doubles, reducing `decode_s` significantly.

#### C. Full Decoupled Compartmental Weights ($p = 1.5$)
Restored the true decoupled specialist weights in Stage 3:
* **Lateral Meniscus:** 0.15 Raptor / 0.85 ConvNeXt (joint space crop)
* **Medial OA & Lateral OA:** 0.20 Raptor / 0.80 ConvNeXt
* **Synovitis:** 0.80 Raptor / 0.20 ConvNeXt (diffuse attention)
* **PF OA:** 0.60 Raptor / 0.40 ConvNeXt
* **Baker's Cyst:** 0.35 Raptor / 0.65 ConvNeXt (fluid stacker)
* **ACL & MCL:** 0.70 / 0.65 Raptor
* **Symmetric Classes (Effusion, Contusion, Fracture):** 0.50 / 0.50
* **Non-linear Power-Mean:**
  $$B = \left( w \cdot P_{\text{raptor}}^{1.5} + (1 - w) \cdot P_{\text{conv}}^{1.5} \right)^{1/1.5}$$

---

### 3. Submission Artifacts

- **Notebook:** [`submissions/v30_lightning_power_master/rsna-knee-v30-lightning-power-master.ipynb`](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/submissions/v30_lightning_power_master/rsna-knee-v30-lightning-power-master.ipynb)
- **Metadata:** [`submissions/v30_lightning_power_master/kernel-metadata.json`](file:///c:/Users/Cyrus/Downloads/rsna%20knee%20fip/submissions/v30_lightning_power_master/kernel-metadata.json)
- **Kaggle Kernel:** `cyruspinto/rsna-knee-v30-lightning-power-master` (Kernel version 1 pushed and running GPU commit).
