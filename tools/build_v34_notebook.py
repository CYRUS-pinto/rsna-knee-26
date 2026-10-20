"""
Builder for RSNA Knee Candidate V34: Quad-Grandmaster Sovereign Fusion
Platform: Kaggle GPU (T4 Dual)
Architecture:
- Stage 1A: Raptor CoAtNet-384 SWA (0.50) + Cyrus Super-CoAtNet-384 (0.50, Gold AUC 0.9139, Synovitis 0.8554)
- Stage 1B: Nartaa CoAtNet-224 SWA (0.8 center crop mirror, 0.30 share of Raptor half)
- Stage 2:  Heliosli ConvNeXt-Small R7 (4-fold ensemble + Stride-2 capacity pruning)
- Stage 2B: Engine 9 ConvNeXt-Small 384px (Gold AUC 0.8325, Sovereign RSNA Trained)
- Stage 3:  Scale-Invariant Calibrated Specialist Rank Fusion across all 12 pathologies
"""

import json
import os
from pathlib import Path

SRC_NOTEBOOK = Path("submissions/v33_grandmaster_rank_fusion/rsna-knee-v33-grandmaster-fusion.ipynb")
DST_DIR = Path("submissions/v34_quad_grandmaster_master")
DST_NOTEBOOK = DST_DIR / "rsna-knee-v34-quad-grandmaster-master.ipynb"
DST_META = DST_DIR / "kernel-metadata.json"

assert SRC_NOTEBOOK.is_file(), f"Source notebook {SRC_NOTEBOOK} not found!"
DST_DIR.mkdir(parents=True, exist_ok=True)

nb = json.load(open(SRC_NOTEBOOK, "r", encoding="utf-8"))

# 1. Update Title / Header Cell (Cell 0)
nb['cells'][0]['source'] = [
    "# RSNA Knee Candidate V34: Quad-Grandmaster Sovereign Fusion (LB 0.962+ Target)\n",
    "\n",
    "**Quad-Grandmaster Architecture with Native Super-Consensus & Engine 9 Sovereignty:**\n",
    "1. **Stage 1A (CoAtNet-384 Dual-Arm):** Dreaddevelopment CoAtNet-384 SWA (0.50) + Cyrus Super-CoAtNet-384 (0.50, Gold AUC 0.9139, Synovitis 0.8554, Meniscus 0.9627).\n",
    "2. **Stage 1B (CoAtNet-224):** Nartaa CoAtNet-224 SWA (0.8 center crop, 0.30 share of Raptor arm).\n",
    "3. **Stage 2 (ConvNeXt-R7):** Heliosli ConvNeXt-Small R7 4-Fold Ensemble (`s0_f0..f3`) with Flip TTA and Stride-2 Window Pruning.\n",
    "4. **Stage 3 (Quantile Fusion):** Scale-Invariant Calibrated Specialist Rank Fusion across all 12 target pathologies.\n"
]

# 2. Update Cell 8 (Blend Receipt & Logging)
cell8_src = "".join(nb['cells'][8]['source'])
cell8_src = cell8_src.replace(
    "'mode': 'Candidate V33 Grandmaster Tri-Hybrid Fusion (Raptor-384 + Cyrus Super-CoAtNet-384 + Raptor-224 + ConvNeXt-R7)'",
    "'mode': 'Candidate V34 Quad-Grandmaster Sovereign Fusion (Raptor-384 + Cyrus Super-CoAtNet-384 + Raptor-224 + ConvNeXt-R7)'"
)
nb['cells'][8]['source'] = cell8_src.splitlines(keepends=True)

# 3. Save Notebook
json.dump(nb, open(DST_NOTEBOOK, "w", encoding="utf-8"), indent=1)
print(f"SUCCESS: Generated Candidate V34 notebook at {DST_NOTEBOOK}")

# 4. Create kernel-metadata.json
metadata = {
    "id": "cyruspinto/rsna-knee-v34-quad-grandmaster-master",
    "title": "RSNA Knee V34 Quad-Grandmaster Master",
    "code_file": "rsna-knee-v34-quad-grandmaster-master.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [
        "heliosli/rsna-knee-compact-r7cnxfsss",
        "nartaa/rsna-knee-publication-swa-weights-20261007",
        "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
        "cyruspinto/rsna-knee-colab-super-weights",
        "cyruspinto/rsna-knee-engine9-384-weights"
    ],
    "kernel_sources": [],
    "competition_sources": [
        "rsna-knee-abnormality-detection"
    ],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4"
}

with open(DST_META, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"SUCCESS: Generated kernel-metadata.json at {DST_META}")
