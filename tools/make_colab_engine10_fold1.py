"""
Builder for Engine 10 Fold 1 Swin-Base-384 Colab Notebook.
Architecture: swin_base_patch4_window12_384.ms_in22k_ft_in1k
Resolution: 384x384px, 2.5D Triplet Windowing
Loss: Asymmetric Focal Loss (gamma=1.5)
Fold: Fold 1 (val_ids = [n_val : 2*n_val])
"""

import json
from pathlib import Path

nb_src = json.load(open("colab_engine10_swin_master.ipynb", "r", encoding="utf-8"))

# Update Title
nb_src["cells"][0]["source"] = [
    "# RSNA Knee Engine 10 Fold 1: Swin-Base-384 2.5D MIL Master (A100 GPU)\n",
    "\n",
    "**Key Architectural Innovations:**\n",
    "- **Hierarchical Vision Transformer:** `swin_base_patch4_window12_384.ms_in22k_ft_in1k` providing pure non-convolutional global attention.\n",
    "- **Zero Inductive Bias Overlap:** Completely orthogonal feature representations from ConvNeXt.\n",
    "- **Fold 1 Training:** Disjoint validation split (`val_ids = shuffled_train[n_val : 2*n_val]`), completing 2-fold Swin representation.\n",
    "- **Asymmetric Focal Loss (gamma=1.5):** Suppresses easy negative slices and targets rare pathologies.\n",
    "- **Target Gold AUC:** >= 0.7800"
]

# Update Fold Split in Training Script (Cell 3)
cell3_src = "".join(nb_src["cells"][3]["source"])
old_split = """    # 5-fold split
    np.random.seed(42)
    shuffled_train = list(train_ids)
    np.random.shuffle(shuffled_train)
    n_val = len(shuffled_train) // 5
    val_ids = shuffled_train[:n_val]
    core_train_ids = shuffled_train[n_val:]"""

new_split = """    # 5-fold split: FOLD 1
    np.random.seed(42)
    shuffled_train = list(train_ids)
    np.random.shuffle(shuffled_train)
    n_val = len(shuffled_train) // 5
    val_ids = shuffled_train[n_val:2*n_val]
    core_train_ids = shuffled_train[:n_val] + shuffled_train[2*n_val:]"""

assert old_split in cell3_src, "Could not find old split in Cell 3!"
cell3_src = cell3_src.replace(old_split, new_split)

# Update checkpoint filename
cell3_src = cell3_src.replace(
    "'engine10_swin_384_best.pt'",
    "'engine10_swin_384_fold1_best.pt'"
)
nb_src["cells"][3]["source"] = cell3_src.splitlines(keepends=True)

# Update Upload Cell (Cell 5)
nb_src["cells"][5]["source"] = [
    "# 5. Package & Publish Engine 10 Fold 1 Weights to Kaggle Datasets\n",
    "import os, json\n",
    "\n",
    "os.makedirs('/content/e10_f1_dataset', exist_ok=True)\n",
    "!cp engine10_swin_384_fold1_best.pt /content/e10_f1_dataset/\n",
    "\n",
    "meta = {\n",
    '    "title": "RSNA Knee Engine 10 Fold 1 Swin-Base Weights",\n',
    '    "id": "cyruspinto/rsna-knee-engine10-fold1-weights",\n',
    '    "licenses": [{"name": "CC0-1.0"}]\n',
    "}\n",
    "with open('/content/e10_f1_dataset/dataset-metadata.json', 'w') as f:\n",
    "    json.dump(meta, f, indent=2)\n",
    "\n",
    "!kaggle datasets create -p /content/e10_f1_dataset -r zip\n"
]

out_nb = Path("colab_engine10_fold1_swin_master.ipynb")
json.dump(nb_src, open(out_nb, "w", encoding="utf-8"), indent=1)
print(f"SUCCESS: Generated {out_nb}")
