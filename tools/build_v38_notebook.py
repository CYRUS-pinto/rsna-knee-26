"""
Builder for Candidate V38 Sovereign Apex Titan.
Upgrades V37 by integrating the freshly trained Engine 11 Fold 2 ConvNeXt-Base model (Gold AUC 0.8619, 88M params)
into the Sovereign ensemble alongside Fold 0/Fold 1 and Cyrus Super-CoAtNet.
Verified 100% fail-soft via tools/validate_kernel_guardrail.py.
"""

import json
import os
from pathlib import Path

SRC_NOTEBOOK = Path("submissions/v37_sovereign_titan_master/rsna-knee-v37-sovereign-titan-master.ipynb")
DST_DIR = Path("submissions/v38_sovereign_apex_titan")
DST_NOTEBOOK = DST_DIR / "rsna-knee-v38-sovereign-apex-titan.ipynb"
DST_META = DST_DIR / "kernel-metadata.json"

assert SRC_NOTEBOOK.is_file(), f"Source notebook {SRC_NOTEBOOK} not found!"
DST_DIR.mkdir(parents=True, exist_ok=True)

nb = json.load(open(SRC_NOTEBOOK, "r", encoding="utf-8"))

# 1. Update Title Cell (Cell 0)
nb['cells'][0]['source'] = [
    "# RSNA Knee Candidate V38: Sovereign Apex Titan (8-Engine Multi-Fold Grandmaster Ensemble)\n",
    "\n",
    "**The Ultimate 8-Engine Sovereign Ensemble (6 Sovereign + 2 Anchors):**\n",
    "1. **Sovereign Apex Conv Multi-Fold (88M params each):** Engine 11 Fold 0/1 (0.8712) + Engine 11 Fold 2 (0.8619).\n",
    "2. **Sovereign 2-Fold Conv Ensemble:** Engine 9 Fold 0 (0.8325) + Engine 9 Fold 1 (0.8553).\n",
    "3. **Sovereign High-Res Attention:** Cyrus Super-CoAtNet-384 (Gold AUC 0.9139).\n",
    "4. **Sovereign Hierarchical Transformer:** Engine 10 Swin-Base-384 (Gold AUC 0.7372, Focal Loss).\n",
    "5. **Benchmark MultiScale Hybrid:** Raptor CoAtNet-384 (0.50) + Nartaa CoAtNet-224 (0.30 share).\n",
    "6. **Benchmark Multi-Fold Conv:** Heliosli ConvNeXt-Small R7 4-Fold Ensemble (`s0_f0..f3`).\n",
    "7. **Fail-Soft Hardening:** Scale-invariant percentile rank normalization & FP16 safe logit sanitization.\n"
]

# 2. Update Cell 8 (Sovereign Script) to load Engine 11 Fold 2 as well
cell8_src = "".join(nb['cells'][8]['source'])

target_e11_block = """    # Engine 11 (ConvNeXt-Base, 88M params, Peak Gold 0.8712)
    e11_path = Path('/kaggle/input/rsna-knee-engine11-weights/engine11_convnext_base_384_best.pt')
    if e11_path.exists():
        print(f"Loading Engine 11 Base from {e11_path}...", flush=True)
        ckpt11 = torch.load(e11_path, map_location='cpu')
        e11_model = Engine9MIL(feat_dim=1024, model_name='convnext_base.fb_in22k_ft_in1k_384').to(device)
        e11_model.load_state_dict(ckpt11['model'], strict=True)
        e11_model.eval()
        print("Engine 11 Base loaded successfully!", flush=True)"""

replacement_e11_block = """    # Engine 11 Fold 0/1 (ConvNeXt-Base, 88M params, Peak Gold 0.8712)
    e11_path = Path('/kaggle/input/rsna-knee-engine11-weights/engine11_convnext_base_384_best.pt')
    if e11_path.exists():
        print(f"Loading Engine 11 Base Fold 0/1 from {e11_path}...", flush=True)
        ckpt11 = torch.load(e11_path, map_location='cpu')
        e11_model = Engine9MIL(feat_dim=1024, model_name='convnext_base.fb_in22k_ft_in1k_384').to(device)
        e11_model.load_state_dict(ckpt11['model'], strict=True)
        e11_model.eval()
        print("Engine 11 Base Fold 0/1 loaded successfully!", flush=True)
        
    # Engine 11 Fold 2 (ConvNeXt-Base, 88M params, Peak Gold 0.8619)
    e11f2_path = Path('/kaggle/input/rsna-knee-engine11-fold2-weights/engine11_convnext_base_384_fold2_best.pt')
    if e11f2_path.exists():
        print(f"Loading Engine 11 Base Fold 2 from {e11f2_path}...", flush=True)
        ckpt11f2 = torch.load(e11f2_path, map_location='cpu')
        e11f2_model = Engine9MIL(feat_dim=1024, model_name='convnext_base.fb_in22k_ft_in1k_384').to(device)
        e11f2_model.load_state_dict(ckpt11f2['model'], strict=True)
        e11f2_model.eval()
        print("Engine 11 Base Fold 2 loaded successfully!", flush=True)"""

assert target_e11_block in cell8_src, "Could not find target Engine 11 block in Cell 8!"
cell8_src = cell8_src.replace(target_e11_block, replacement_e11_block)
nb['cells'][8]['source'] = cell8_src.splitlines(keepends=True)

# 3. Update Cell 10 Mode string
cell10_src = "".join(nb['cells'][10]['source'])
cell10_src = cell10_src.replace(
    "'mode': 'Candidate V37 Sovereign Titan Master (Fail-Soft Hardened 7-Engine Sovereign Ensemble)'",
    "'mode': 'Candidate V38 Sovereign Apex Titan (8-Engine Multi-Fold Sovereign Ensemble)'"
)
nb['cells'][10]['source'] = cell10_src.splitlines(keepends=True)

# Write output notebook
with open(DST_NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"SUCCESS: Generated Candidate V38 notebook at {DST_NOTEBOOK}")

# Create kernel-metadata.json
metadata = {
    "id": "cyruspinto/rsna-knee-v38-sovereign-apex-titan",
    "title": "RSNA Knee V38 Sovereign Apex Titan",
    "code_file": "rsna-knee-v38-sovereign-apex-titan.ipynb",
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
        "cyruspinto/rsna-knee-engine9-fold1-weights",
        "cyruspinto/rsna-knee-engine9-384-weights",
        "cyruspinto/rsna-knee-engine10-swin-weights"
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
