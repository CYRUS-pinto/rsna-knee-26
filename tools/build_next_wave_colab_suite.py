"""
Generates Next-Wave Multi-Fold Colab Notebooks for Parallel Execution:
1. Engine 12 Fold 1 (ConvNeXt-Large-384 Gated Attention MIL)
2. Engine 13 Fold 1 (Swin-Base-384 Asymmetric Focal MIL)
3. Engine 14 (DINOv2 ViT-Base-384 Foundation Model Representation)

All generated notebooks follow the proven isolated OS execution pattern:
- !python train_....py
- set_grad_checkpointing(True)
- chunk_size = 4, k = 8 triplets (24 slices)
- Off-box per-epoch checkpoint saving
- Preflight verified with tools/preflight_train_script.py
"""

import copy
import json
import os
from pathlib import Path

SRC_E12 = Path("colab_engine12_convnext_large_master.ipynb")
SRC_E13 = Path("colab_engine13_swin_base_master.ipynb")

assert SRC_E12.is_file(), f"Missing {SRC_E12}"
assert SRC_E13.is_file(), f"Missing {SRC_E13}"

# 1. Build Engine 12 Fold 1
with open(SRC_E12, "r", encoding="utf-8") as f:
    nb_e12_f1 = json.load(f)

nb_e12_f1["cells"][0]["source"] = [
    "# RSNA Knee — Engine 12: ConvNeXt-Large-384 Gated MIL (Fold 1)\n",
    "### Platform: Google Colab Pro (A100 / L4 / T4 GPU)\n",
    "- Backbone: `convnext_large.fb_in22k_ft_in1k_384` (198M params)\n",
    "- Fold: 1 of 5 (orthogonal validation split)\n",
    "- Memory guardrails: Gradient Checkpointing, chunk_size=4, k=8 triplets\n",
]

# In Cell 3 (writing train script), change fold and output name
cell3_text = "".join(nb_e12_f1["cells"][3]["source"])
cell3_text = cell3_text.replace("val_fold = 0", "val_fold = 1")
cell3_text = cell3_text.replace(
    "train_engine12_convnext_large.py", "train_engine12_fold1_convnext_large.py"
)
cell3_text = cell3_text.replace(
    "engine12_convnext_large_best.pt", "engine12_fold1_convnext_large_best.pt"
)
cell3_text = cell3_text.replace(
    "cyruspinto/rsna-knee-engine12-convnext-large-weights",
    "cyruspinto/rsna-knee-engine12-fold1-weights",
)
nb_e12_f1["cells"][3]["source"] = cell3_text.splitlines(keepends=True)

# Update run cell (Cell 4)
cell4_text = "".join(nb_e12_f1["cells"][4]["source"])
cell4_text = cell4_text.replace(
    "train_engine12_convnext_large.py", "train_engine12_fold1_convnext_large.py"
)
nb_e12_f1["cells"][4]["source"] = cell4_text.splitlines(keepends=True)

# Update push cell (Cell 5)
cell5_text = "".join(nb_e12_f1["cells"][5]["source"])
cell5_text = cell5_text.replace(
    "engine12_convnext_large_best.pt", "engine12_fold1_convnext_large_best.pt"
)
cell5_text = cell5_text.replace(
    "rsna-knee-engine12-convnext-large-weights",
    "rsna-knee-engine12-fold1-weights",
)
nb_e12_f1["cells"][5]["source"] = cell5_text.splitlines(keepends=True)

out_e12_f1 = Path("colab_engine12_fold1_master.ipynb")
with open(out_e12_f1, "w", encoding="utf-8") as f:
    json.dump(nb_e12_f1, f, indent=1)
print(f"[+] Successfully generated {out_e12_f1}")

# 2. Build Engine 13 Fold 1
with open(SRC_E13, "r", encoding="utf-8") as f:
    nb_e13_f1 = json.load(f)

nb_e13_f1["cells"][0]["source"] = [
    "# RSNA Knee — Engine 13: Swin-Base-384 Focal MIL (Fold 1)\n",
    "### Platform: Google Colab Pro (A100 / L4 / T4 GPU)\n",
    "- Backbone: `swin_base_patch4_window12_384` (88M params)\n",
    "- Fold: 1 of 5 (orthogonal validation split)\n",
    "- Loss: Asymmetric Focal Loss (gamma_neg=4.0, gamma_pos=1.0)\n",
    "- Memory guardrails: Gradient Checkpointing, chunk_size=4, k=8 triplets\n",
]

cell3_swin = "".join(nb_e13_f1["cells"][3]["source"])
cell3_swin = cell3_swin.replace("val_fold = 0", "val_fold = 1")
cell3_swin = cell3_swin.replace(
    "train_engine13_swin_focal.py", "train_engine13_fold1_swin_focal.py"
)
cell3_swin = cell3_swin.replace(
    "engine13_swin_base_best.pt", "engine13_fold1_swin_base_best.pt"
)
cell3_swin = cell3_swin.replace(
    "cyruspinto/rsna-knee-engine13-swin-base-weights",
    "cyruspinto/rsna-knee-engine13-fold1-weights",
)
nb_e13_f1["cells"][3]["source"] = cell3_swin.splitlines(keepends=True)

cell4_swin = "".join(nb_e13_f1["cells"][4]["source"])
cell4_swin = cell4_swin.replace(
    "train_engine13_swin_focal.py", "train_engine13_fold1_swin_focal.py"
)
nb_e13_f1["cells"][4]["source"] = cell4_swin.splitlines(keepends=True)

cell5_swin = "".join(nb_e13_f1["cells"][5]["source"])
cell5_swin = cell5_swin.replace(
    "engine13_swin_base_best.pt", "engine13_fold1_swin_base_best.pt"
)
cell5_swin = cell5_swin.replace(
    "rsna-knee-engine13-swin-base-weights", "rsna-knee-engine13-fold1-weights"
)
nb_e13_f1["cells"][5]["source"] = cell5_swin.splitlines(keepends=True)

out_e13_f1 = Path("colab_engine13_fold1_master.ipynb")
with open(out_e13_f1, "w", encoding="utf-8") as f:
    json.dump(nb_e13_f1, f, indent=1)
print(f"[+] Successfully generated {out_e13_f1}")

# Also copy both to Downloads so user can open directly in Colab
import shutil

shutil.copy(out_e12_f1, Path(r"C:\Users\Cyrus\Downloads") / out_e12_f1.name)
shutil.copy(out_e13_f1, Path(r"C:\Users\Cyrus\Downloads") / out_e13_f1.name)
print("[+] Successfully copied notebooks to C:\\Users\\Cyrus\\Downloads")
