"""
Builder for Engine 11 Fold 3: ConvNeXt-Base 384px High-Capacity MIL (A100 Master).
Backbone: convnext_base.fb_in22k_ft_in1k_384 (88M params, feat_dim=1024)
Fold: Fold 3 (val_ids = [3*n_val : 4*n_val], core_train = [:3*n_val] + [4*n_val:])
Target: Gold AUC >= 0.8700
"""

import json
from pathlib import Path

nb_src = json.load(open("colab_engine11_fold2_master.ipynb", "r", encoding="utf-8"))

# 1. Update Title Cell (Cell 0)
nb_src["cells"][0]["source"] = [
    "# Autonomous Engine 11 Fold 3: ConvNeXt-Base 384px High-Capacity MIL (A100 Master)\n",
    "\n",
    "**Objective:** Complete Fold 3 of the 88M-parameter Apex ConvNeXt-Base backbone to achieve a multi-fold sovereign ensemble.\n",
    "- **Architecture:** `convnext_base.fb_in22k_ft_in1k_384` (88M parameters, feat_dim=1024)\n",
    "- **Resolution:** 384x384px, 2.5D Triplet Windowing\n",
    "- **Folds:** Fold 3 (`val_ids = shuffled_train[3*n_val : 4*n_val]`)\n",
    "- **Training Time:** ~18 minutes on Colab A100 (~0.6 compute units)\n",
    "- **Target Gold AUC:** >= 0.8700"
]

# 2. Update Training Script Cell (Cell 3)
cell3_src = "".join(nb_src["cells"][3]["source"])

# Replace fold split
old_split = """    # 5-fold split: FOLD 2 (disjoint from Fold 0 & Fold 1)
    np.random.seed(42)
    shuffled_train = list(train_ids)
    np.random.shuffle(shuffled_train)
    n_val = len(shuffled_train) // 5
    val_ids = shuffled_train[2*n_val:3*n_val]
    core_train_ids = shuffled_train[:2*n_val] + shuffled_train[3*n_val:]
    print(f'Fold 2 Split: {len(core_train_ids)} train, {len(val_ids)} validation.', flush=True)"""

new_split = """    # 5-fold split: FOLD 3 (disjoint from Fold 0, 1 & 2)
    np.random.seed(42)
    shuffled_train = list(train_ids)
    np.random.shuffle(shuffled_train)
    n_val = len(shuffled_train) // 5
    val_ids = shuffled_train[3*n_val:4*n_val]
    core_train_ids = shuffled_train[:3*n_val] + shuffled_train[4*n_val:]
    print(f'Fold 3 Split: {len(core_train_ids)} train, {len(val_ids)} validation.', flush=True)"""

assert old_split in cell3_src, "Could not find old fold split in Cell 3!"
cell3_src = cell3_src.replace(old_split, new_split)

# Replace checkpoint names
cell3_src = cell3_src.replace("train_engine11_fold2.py", "train_engine11_fold3.py")
cell3_src = cell3_src.replace("engine11_convnext_base_384_fold2_best.pt", "engine11_convnext_base_384_fold3_best.pt")
cell3_src = cell3_src.replace("Fold 2", "Fold 3")
nb_src["cells"][3]["source"] = cell3_src.splitlines(keepends=True)

# 3. Update Launch Cell (Cell 4)
nb_src["cells"][4]["source"] = ["!python train_engine11_fold3.py\n"]

# 4. Update Dataset Packaging Cell (Cell 5)
nb_src["cells"][5]["source"] = [
    "# 5. Package & Publish Fold 3 Weights to Kaggle Datasets\n",
    "import os, json\n",
    "\n",
    "os.makedirs('/content/e11_f3_dataset', exist_ok=True)\n",
    "!cp engine11_convnext_base_384_fold3_best.pt /content/e11_f3_dataset/\n",
    "\n",
    "meta = {\n",
    '    "title": "RSNA Knee Engine 11 Fold 3 ConvNeXt-Base Weights",\n',
    '    "id": "cyruspinto/rsna-knee-engine11-fold3-weights",\n',
    '    "licenses": [{"name": "CC0-1.0"}]\n',
    "}\n",
    "with open('/content/e11_f3_dataset/dataset-metadata.json', 'w') as f:\n",
    "    json.dump(meta, f, indent=2)\n",
    "\n",
    "!kaggle datasets create -p /content/e11_f3_dataset -r zip\n"
]

out_nb = Path("colab_engine11_fold3_master.ipynb")
json.dump(nb_src, open(out_nb, "w", encoding="utf-8"), indent=1)
print(f"SUCCESS: Generated {out_nb}")
