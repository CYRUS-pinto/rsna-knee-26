"""
Builder for Candidate V41 Sovereign Hendecet Titan.
Integrates ALL 9 Sovereign Models + 2 Benchmark Anchors (11-Engine Championship Ensemble):
1. Cyrus Super-CoAtNet-384 (Gold AUC 0.9139)
2. Engine 11 Fold 0/1 ConvNeXt-Base (Gold AUC 0.8712, 88M params)
3. Engine 11 Fold 2 ConvNeXt-Base (Gold AUC 0.8619, 88M params)
4. Engine 11 Fold 3 ConvNeXt-Base (Gold AUC 0.8561, 88M params)
5. Engine 11 Fold 4 ConvNeXt-Base (Gold AUC 0.8506, 88M params, NEW)
6. Engine 10 Fold 1 Swin-Base-384 Vision Transformer (Gold AUC 0.8634)
7. Engine 10 Fold 0 Swin-Base-384 Vision Transformer (Gold AUC 0.7372)
8. Engine 9 Fold 1 ConvNeXt-Small 384px (Gold AUC 0.8553)
9. Engine 9 Fold 0 ConvNeXt-Small 384px (Gold AUC 0.8325)
10. Raptor CoAtNet MultiScale (384 + 224)
11. Heliosli ConvNeXt-Small R7 4-Fold Ensemble
"""

import json
import os
from pathlib import Path

SRC_NOTEBOOK = Path("submissions/v40_sovereign_decet_titan/rsna-knee-v40-sovereign-decet-titan.ipynb")
DST_DIR = Path("submissions/v41_sovereign_hendecet_titan")
DST_NOTEBOOK = DST_DIR / "rsna-knee-v41-sovereign-hendecet-titan.ipynb"
DST_META = DST_DIR / "kernel-metadata.json"

assert SRC_NOTEBOOK.is_file(), f"Source notebook {SRC_NOTEBOOK} not found!"
DST_DIR.mkdir(parents=True, exist_ok=True)

nb = json.load(open(SRC_NOTEBOOK, "r", encoding="utf-8"))

# 1. Update Title Cell (Cell 0)
nb['cells'][0]['source'] = [
    "# RSNA Knee Candidate V41: Sovereign Hendecet Titan (11-Engine Full Cross-Fold Grandmaster Ensemble)\n",
    "\n",
    "**The Apex 11-Engine Championship Architecture (9 Sovereign + 2 Anchors):**\n",
    "1. **Sovereign Apex Conv FULL 4-Fold Ensemble (88M params each):** Engine 11 Fold 0/1 (0.8712) + Fold 2 (0.8619) + Fold 3 (0.8561) + Fold 4 (0.8506).\n",
    "2. **Sovereign Hierarchical Transformer 2-Fold:** Engine 10 Fold 1 (0.8634) + Engine 10 Fold 0 (0.7372).\n",
    "3. **Sovereign High-Res Attention:** Cyrus Super-CoAtNet-384 (Gold AUC 0.9139).\n",
    "4. **Sovereign 2-Fold Conv Ensemble:** Engine 9 Fold 0 (0.8325) + Engine 9 Fold 1 (0.8553).\n",
    "5. **Benchmark MultiScale Hybrid:** Raptor CoAtNet-384 (0.50) + Nartaa CoAtNet-224 (0.30 share).\n",
    "6. **Benchmark Multi-Fold Conv:** Heliosli ConvNeXt-Small R7 4-Fold Ensemble (`s0_f0..f3`).\n",
    "7. **Fail-Soft Hardening:** Scale-invariant percentile rank normalization & FP16 safe logit sanitization.\n"
]

# 2. Update Cell 8 (Sovereign Script) to load Engine 11 Fold 4 as well
cell8_src = "".join(nb['cells'][8]['source'])

target_e11_f3_block = """    # Engine 11 Fold 3 (ConvNeXt-Base, 88M params, Peak Gold 0.8561)
    e11f3_path = Path('/kaggle/input/rsna-knee-engine11-fold3-weights/engine11_convnext_base_384_fold3_best.pt')
    if e11f3_path.exists():
        print(f"Loading Engine 11 Base Fold 3 from {e11f3_path}...", flush=True)
        ckpt11f3 = torch.load(e11f3_path, map_location='cpu')
        e11f3_model = Engine9MIL(feat_dim=1024, model_name='convnext_base.fb_in22k_ft_in1k_384').to(device)
        e11f3_model.load_state_dict(ckpt11f3['model'], strict=True)
        e11f3_model.eval()
        print("Engine 11 Base Fold 3 loaded successfully!", flush=True)"""

replacement_e11_f3_f4_block = """    # Engine 11 Fold 3 (ConvNeXt-Base, 88M params, Peak Gold 0.8561)
    e11f3_path = Path('/kaggle/input/rsna-knee-engine11-fold3-weights/engine11_convnext_base_384_fold3_best.pt')
    if e11f3_path.exists():
        print(f"Loading Engine 11 Base Fold 3 from {e11f3_path}...", flush=True)
        ckpt11f3 = torch.load(e11f3_path, map_location='cpu')
        e11f3_model = Engine9MIL(feat_dim=1024, model_name='convnext_base.fb_in22k_ft_in1k_384').to(device)
        e11f3_model.load_state_dict(ckpt11f3['model'], strict=True)
        e11f3_model.eval()
        print("Engine 11 Base Fold 3 loaded successfully!", flush=True)
        
    # Engine 11 Fold 4 (ConvNeXt-Base, 88M params, Peak Gold 0.8506)
    e11f4_path = Path('/kaggle/input/rsna-knee-engine11-fold4-weights/engine11_convnext_base_384_fold4_best.pt')
    if e11f4_path.exists():
        print(f"Loading Engine 11 Base Fold 4 from {e11f4_path}...", flush=True)
        ckpt11f4 = torch.load(e11f4_path, map_location='cpu')
        e11f4_model = Engine9MIL(feat_dim=1024, model_name='convnext_base.fb_in22k_ft_in1k_384').to(device)
        e11f4_model.load_state_dict(ckpt11f4['model'], strict=True)
        e11f4_model.eval()
        print("Engine 11 Base Fold 4 loaded successfully!", flush=True)"""

assert target_e11_f3_block in cell8_src, "Could not find target Engine 11 Fold 3 block in Cell 8!"
cell8_src = cell8_src.replace(target_e11_f3_block, replacement_e11_f3_f4_block)
nb['cells'][8]['source'] = cell8_src.splitlines(keepends=True)

# 3. Update Cell 10 Mode string
cell10_src = "".join(nb['cells'][10]['source'])
cell10_src = cell10_src.replace(
    "'mode': 'Candidate V40 Sovereign Decet Titan (10-Engine Multi-Fold Sovereign Ensemble)'",
    "'mode': 'Candidate V41 Sovereign Hendecet Titan (11-Engine Full Cross-Fold Sovereign Ensemble)'"
)
nb['cells'][10]['source'] = cell10_src.splitlines(keepends=True)

# Write output notebook
with open(DST_NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"SUCCESS: Generated Candidate V41 notebook at {DST_NOTEBOOK}")

# Create kernel-metadata.json
metadata = {
    "id": "cyruspinto/rsna-knee-v41-sovereign-hendecet-titan",
    "title": "RSNA Knee V41 Sovereign Hendecet Titan",
    "code_file": "rsna-knee-v41-sovereign-hendecet-titan.ipynb",
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
        "cyruspinto/rsna-knee-engine11-weights",
        "cyruspinto/rsna-knee-engine11-fold2-weights",
        "cyruspinto/rsna-knee-engine11-fold3-weights",
        "cyruspinto/rsna-knee-engine11-fold4-weights",
        "cyruspinto/rsna-knee-engine10-fold1-weights",
        "cyruspinto/rsna-knee-engine10-swin-weights",
        "cyruspinto/rsna-knee-engine9-fold1-weights",
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
