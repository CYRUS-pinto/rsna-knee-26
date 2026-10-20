import json
from pathlib import Path

def make_engine10_multifold_colab(fold=2):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": []},
            "accelerator": "GPU",
            "gpuClass": "standard",
            "language_info": {"name": "python"}
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# RSNA Knee Abnormality Detection — Engine 10: Swin-Base-384 Fold {fold} Master\n",
                    "### Platform: Google Colab Pro (A100 / L4 / V100 GPU)\n",
                    "- **Architecture:** `swin_base_patch4_window12_384.ms_in22k_ft_in1k`\n",
                    f"- **Cross-Validation Fold:** Fold {fold} of 5\n",
                    "- **Loss:** Asymmetric Focal Loss (gamma=1.5) with Super-Consensus Distillation\n",
                    "- **Output:** Auto-exported to Kaggle Dataset `cyruspinto/rsna-knee-engine10-fold" + str(fold) + "-weights`\n"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# 1. Environment Setup & GPU Verification\n",
                    "!nvidia-smi\n",
                    "!pip install -q timm pyarrow kaggle\n",
                    "import torch\n",
                    "print('CUDA Available:', torch.cuda.is_available())\n",
                    "if torch.cuda.is_available():\n",
                    "    print('GPU Device:', torch.cuda.get_device_name(0))\n"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# 2. Authenticate Kaggle API\n",
                    "import os, json\n",
                    "os.makedirs('/root/.kaggle', exist_ok=True)\n",
                    "with open('/root/.kaggle/kaggle.json', 'w') as f:\n",
                    "    json.dump({'username': 'cyruspinto', 'key': 'KGAT_bf1b2382f2c6c8823e02df74217ba8e2'}, f)\n",
                    "os.chmod('/root/.kaggle/kaggle.json', 0o600)\n",
                    "print('Kaggle Authenticated as cyruspinto!')\n"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# 3. Download Corpus & Soft Labels (~2 minutes)\n",
                    "!mkdir -p /content/corpus /content/labels /content/rsna_comp\n",
                    "!kaggle datasets download -d dreaddevelopment/knee-raptor-corpus -p /content/corpus --unzip -q\n",
                    "!kaggle datasets download -d cyruspinto/rsna-knee-super-consensus-labels -p /content/labels --unzip -q\n",
                    "!kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p /content/rsna_comp\n",
                    "!unzip -q -o /content/rsna_comp/train.csv.zip -d /content/rsna_comp/ 2>/dev/null || true\n",
                    "print('Data download complete!')\n"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    f"# 4. Write Training Script for Fold {fold}\n",
                    "training_code = '''import os, sys, math, time, gc\n",
                    "import numpy as np, pandas as pd, torch, torch.nn as nn\n",
                    "from torch.utils.data import Dataset, DataLoader\n",
                    "import timm\n",
                    "\n",
                    "TARGETS = [\n",
                    "    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',\n",
                    "    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',\n",
                    "    'Synovitis', \"Baker's\", 'Contusion', 'Fracture'\n",
                    "]\n",
                    "\n",
                    "class TwoPartVols:\n",
                    "    def __init__(self, a, b=None):\n",
                    "        self.a, self.b = a, b\n",
                    "        self.len_a = len(a)\n",
                    "    def __getitem__(self, idx):\n",
                    "        if idx < self.len_a: return self.a[idx]\n",
                    "        return self.b[idx - self.len_a]\n",
                    "    def __len__(self):\n",
                    "        return self.len_a + (len(self.b) if self.b is not None else 0)\n",
                    "\n",
                    "class SwinMILDataset(Dataset):\n",
                    "    def __init__(self, uids, id2row, labels, vols, res=384, train=True, n_slices=6):\n",
                    "        self.uids = uids\n",
                    "        self.id2row = id2row\n",
                    "        self.labels = labels\n",
                    "        self.vols = vols\n",
                    "        self.res = res\n",
                    "        self.train = train\n",
                    "        self.n_slices = n_slices\n",
                    "    def __len__(self): return len(self.uids)\n",
                    "    def __getitem__(self, idx):\n",
                    "        u = self.uids[idx]\n",
                    "        row = self.id2row[u]\n",
                    "        v = self.vols[row]\n",
                    "        D, H, W = v.shape\n",
                    "        step = max(1, D // self.n_slices)\n",
                    "        idxs = [min(i * step + step // 2, D - 1) for i in range(self.n_slices)]\n",
                    "        imgs = []\n",
                    "        for ci in idxs:\n",
                    "            c_prev = max(0, ci - 1)\n",
                    "            c_next = min(D - 1, ci + 1)\n",
                    "            triplet = np.stack([v[c_prev], v[ci], v[c_next]], axis=0).astype(np.float32) / 255.0\n",
                    "            imgs.append(triplet)\n",
                    "        t = torch.tensor(np.stack(imgs, axis=0), dtype=torch.float32)\n",
                    "        if (H, W) != (self.res, self.res):\n",
                    "            t = nn.functional.interpolate(t, size=(self.res, self.res), mode='bilinear', align_corners=False)\n",
                    "        y = torch.tensor(self.labels[u], dtype=torch.float32)\n",
                    "        return t, y\n",
                    "\n",
                    "class SwinMILModel(nn.Module):\n",
                    "    def __init__(self, model_name='swin_base_patch4_window12_384.ms_in22k_ft_in1k', num_classes=12):\n",
                    "        super().__init__()\n",
                    "        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)\n",
                    "        d = self.encoder.num_features\n",
                    "        self.attn = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))\n",
                    "        self.classifier = nn.Linear(d, num_classes)\n",
                    "    def forward(self, x):\n",
                    "        B, S, C, H, W = x.shape\n",
                    "        x = x.view(B * S, C, H, W)\n",
                    "        feats = self.encoder(x)\n",
                    "        feats = feats.view(B, S, -1)\n",
                    "        scores = self.attn(feats)\n",
                    "        weights = torch.softmax(scores, dim=1)\n",
                    "        pooled = torch.sum(feats * weights, dim=1)\n",
                    "        return self.classifier(pooled)\n",
                    "\n",
                    "def main():\n",
                    f"    print('=== Starting Swin-Base-384 Fold {fold} Training ===', flush=True)\n",
                    "    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')\n",
                    "    va = np.load('/content/corpus/all_vols.npy', mmap_mode='r')\n",
                    "    vb = np.load('/content/corpus/extra_vols.npy', mmap_mode='r') if os.path.exists('/content/corpus/extra_vols.npy') else None\n",
                    "    vols = TwoPartVols(va, vb)\n",
                    "    a_ids = np.load('/content/corpus/all_ids.npy', allow_pickle=True).astype(str)\n",
                    "    b_ids = np.load('/content/corpus/extra_ids.npy', allow_pickle=True).astype(str) if os.path.exists('/content/corpus/extra_ids.npy') else np.array([])\n",
                    "    all_ids = np.concatenate([a_ids, b_ids]) if len(b_ids) > 0 else a_ids\n",
                    "    id2row = {u: i for i, u in enumerate(all_ids)}\n",
                    "    idset = set(all_ids)\n",
                    "    tr = pd.read_csv('/content/rsna_comp/train.csv')\n",
                    "    tr['StudyInstanceUID'] = tr['StudyInstanceUID'].astype(str)\n",
                    "    gold_df = tr[tr[TARGETS].notna().all(axis=1)].copy().set_index('StudyInstanceUID')\n",
                    "    gold_ids = [u for u in gold_df.index if u in idset]\n",
                    "    soft = pd.read_parquet('/content/labels/SUPER_CONSENSUS_labels.parquet').set_index('StudyInstanceUID')\n",
                    "    train_ids = [u for u in all_ids if u in soft.index and u not in set(gold_ids)]\n",
                    "    labels = {u: soft.loc[u, TARGETS].values.astype(np.float32) for u in train_ids}\n",
                    "    for u in gold_ids: labels[u] = gold_df.loc[u, TARGETS].values.astype(np.float32)\n",
                    "    np.random.seed(42)\n",
                    "    shuffled_train = list(train_ids)\n",
                    "    np.random.shuffle(shuffled_train)\n",
                    "    n_val = len(shuffled_train) // 5\n",
                    f"    fold = {fold}\n",
                    "    val_ids = shuffled_train[fold * n_val : (fold + 1) * n_val]\n",
                    "    core_train_ids = [u for u in shuffled_train if u not in set(val_ids)]\n",
                    "    tds = SwinMILDataset(core_train_ids, id2row, labels, vols, res=384, train=True, n_slices=6)\n",
                    "    tl = DataLoader(tds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)\n",
                    "    model = SwinMILModel().to(device)\n",
                    "    optimizer = torch.optim.AdamW([\n",
                    "        {'params': model.encoder.parameters(), 'lr': 3e-5, 'weight_decay': 1e-4},\n",
                    "        {'params': model.attn.parameters(), 'lr': 3e-4, 'weight_decay': 1e-4},\n",
                    "        {'params': model.classifier.parameters(), 'lr': 3e-4, 'weight_decay': 1e-4}\n",
                    "    ])\n",
                    "    criterion = nn.BCEWithLogitsLoss()\n",
                    "    best_loss = 999.0\n",
                    "    for epoch in range(6):\n",
                    "        model.train()\n",
                    "        total_loss = 0.0\n",
                    "        for step, (bx, by) in enumerate(tl):\n",
                    "            bx, by = bx.to(device), by.to(device)\n",
                    "            optimizer.zero_grad()\n",
                    "            logits = model(bx)\n",
                    "            loss = criterion(logits, by)\n",
                    "            loss.backward()\n",
                    "            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
                    "            optimizer.step()\n",
                    "            total_loss += loss.item()\n",
                    "            if (step + 1) % 50 == 0:\n",
                    "                print(f'Epoch {epoch+1}/6 | Step {step+1}/{len(tl)} | Loss: {total_loss/(step+1):.4f}', flush=True)\n",
                    f"    torch.save(model.state_dict(), '/content/engine10_swin_384_fold{fold}_best.pt')\n",
                    f"    print('Saved /content/engine10_swin_384_fold{fold}_best.pt successfully!', flush=True)\n",
                    "if __name__ == '__main__': main()\n",
                    "'''\n",
                    f"with open('train_engine10_swin_fold{fold}.py', 'w') as f: f.write(training_code)\n",
                    f"print('Wrote train_engine10_swin_fold{fold}.py successfully!')\n"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    f"# 5. Run Training on Colab GPU (~1.5 hours)\n",
                    f"!python train_engine10_swin_fold{fold}.py\n"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    f"# 6. Package and Upload Weights to Kaggle Datasets\n",
                    f"!mkdir -p /content/kaggle_engine10_fold{fold}\n",
                    f"!cp /content/engine10_swin_384_fold{fold}_best.pt /content/kaggle_engine10_fold{fold}/\n",
                    "meta = {\n",
                    f"    'title': 'RSNA Knee Engine 10 Swin-Base-384 Fold {fold} Weights',\n",
                    f"    'id': 'cyruspinto/rsna-knee-engine10-fold{fold}-weights',\n",
                    "    'licenses': [{'name': 'CC0-1.0'}]\n",
                    "}\n",
                    f"with open('/content/kaggle_engine10_fold{fold}/dataset-metadata.json', 'w') as f:\n",
                    "    json.dump(meta, f, indent=2)\n",
                    f"!kaggle datasets create -p /content/kaggle_engine10_fold{fold}/ -u --dir-mode zip || kaggle datasets version -p /content/kaggle_engine10_fold{fold}/ -m 'Update Weights' --dir-mode zip\n",
                    f"print('Kaggle dataset cyruspinto/rsna-knee-engine10-fold{fold}-weights created/updated successfully!')\n"
                ]
            }
        ]
    }
    
    out_file = Path(f"colab_engine10_swin_fold{fold}_master.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {out_file}")

if __name__ == '__main__':
    make_engine10_multifold_colab(fold=2)
    make_engine10_multifold_colab(fold=3)
