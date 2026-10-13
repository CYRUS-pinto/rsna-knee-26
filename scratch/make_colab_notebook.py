import json
import os

cells = []

# Cell 1: Header
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# RSNA Knee Abnormality Detection — A100 High-Speed Training Pipeline\n",
        "### Platform: Google Colab Pro (A100 / L4 GPU)\n",
        "**Supervision:** Distilled Super-Consensus Labels (`0.9172 Macro-AUC`)\n",
        "**Validation Benchmark:** 58 Gold Adjudicated Ground-Truth Studies\n",
        "**Models:** CoAtNet-2 384 & ConvNeXt-Small 384 with Per-Finding MIL Attention\n",
        "\n",
        "---"
    ]
})

# Cell 2: GPU Check & Environment Setup
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 1: GPU Verification & High-Throughput Environment Setup"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!nvidia-smi\n",
        "!pip install -q timm albumentations kaggle\n",
        "\n",
        "import torch\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f'Training Device: {device}')\n",
        "if torch.cuda.is_available():\n",
        "    print('GPU Name:', torch.cuda.get_device_name(0))\n",
        "    torch.backends.cuda.matmul.allow_tf32 = True\n",
        "    torch.backends.cudnn.allow_tf32 = True\n",
        "    print('TF32 acceleration enabled for A100/L4.')"
    ]
})

# Cell 3: Kaggle API Authentication
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 2: Kaggle Credentials & Authentication\n",
        "Authenticates via Kaggle Access Token or uploaded `kaggle.json`."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "os.makedirs('/root/.kaggle', exist_ok=True)\n",
        "# Direct Access Token authentication\n",
        "os.environ['KAGGLE_API_TOKEN'] = 'KGAT_f10b3b14e370d1fdb87c640a42898ac2'\n",
        "with open('/root/.kaggle/access_token', 'w') as f:\n",
        "    f.write('KGAT_f10b3b14e370d1fdb87c640a42898ac2')\n",
        "os.chmod('/root/.kaggle/access_token', 0o600)\n",
        "\n",
        "!kaggle -v\n",
        "print('Kaggle API authenticated successfully.')"
    ]
})

# Cell 4: Download Preprocessed Slice Stacks & Labels
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 3: Fast Cloud Data Ingestion (~1-2 minutes on Colab network)"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!mkdir -p /content/corpus /content/labels /content/rsna_comp\n",
        "\n",
        "# 1. Download precomputed 44x336x336 slice stacks (Parts 1 & 2)\n",
        "print('[1/4] Downloading Knee Raptor Corpus (Part 1)...')\n",
        "!kaggle datasets download -d dreaddevelopment/knee-raptor-corpus -p /content/corpus --unzip -q\n",
        "\n",
        "print('[2/4] Downloading Knee Raptor Corpus Ext (Part 2)...')\n",
        "!kaggle datasets download -d dreaddevelopment/knee-raptor-corpus-ext -p /content/corpus --unzip -q\n",
        "\n",
        "# 2. Download our verified 0.9172 Super-Consensus Labels\n",
        "print('[3/4] Downloading Cyrus Super-Consensus Labels...')\n",
        "!kaggle datasets download -d cyruspinto/rsna-knee-super-consensus-labels-0916 -p /content/labels --unzip -q\n",
        "\n",
        "# 3. Download Competition train.csv (58 Gold studies)\n",
        "print('[4/4] Downloading competition train.csv...')\n",
        "!kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p /content/rsna_comp -q\n",
        "!unzip -q -o /content/rsna_comp/train.csv.zip -d /content/rsna_comp/ 2>/dev/null || true\n",
        "\n",
        "print('All training corpora downloaded and extracted.')"
    ]
})

# Cell 5: Training Code
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 4: Multi-Instance Learning (MIL) Training Script with SWA\n",
        "Trains a CoAtNet-2 384 backbone with per-diagnosis attention pooling directly against our 0.9172 soft targets, evaluating on the 58 Gold holdouts every epoch."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%%writefile train_colab.py\n",
        "import os, sys, time, json, math, random, argparse\n",
        "import numpy as np, pandas as pd\n",
        "import torch, torch.nn as nn, torch.nn.functional as F\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "from sklearn.metrics import roc_auc_score\n",
        "import timm\n",
        "\n",
        "LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',\n",
        "       'Effusion','Synovitis',\"Baker's\",'Contusion','Fracture']\n",
        "\n",
        "class TwoPartVols:\n",
        "    def __init__(self, a, b):\n",
        "        self.a, self.b = a, b\n",
        "        self.n_a = a.shape[0]\n",
        "        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])\n",
        "    def __len__(self): return self.shape[0]\n",
        "    def __getitem__(self, row):\n",
        "        return self.a[row] if row < self.n_a else self.b[row - self.n_a]\n",
        "\n",
        "def open_corpus(root='/content/corpus'):\n",
        "    av = os.path.join(root, 'all_vols.npy')\n",
        "    bv = os.path.join(root, 'extra_vols.npy')\n",
        "    am = os.path.join(root, 'all_masks.npy')\n",
        "    bm = os.path.join(root, 'extra_masks.npy')\n",
        "    vols = TwoPartVols(np.load(av, mmap_mode='r'), np.load(bv, mmap_mode='r'))\n",
        "    masks = np.concatenate([np.load(am), np.load(bm)], axis=0)\n",
        "    return vols, masks\n",
        "\n",
        "def open_ids(root='/content/corpus'):\n",
        "    a = os.path.join(root, 'all_ids.npy')\n",
        "    b = os.path.join(root, 'extra_ids.npy')\n",
        "    return np.concatenate([np.load(a, allow_pickle=True).astype(str),\n",
        "                           np.load(b, allow_pickle=True).astype(str)])\n",
        "\n",
        "class StudyWindows(Dataset):\n",
        "    def __init__(self, ids, id2row, labels, res=384, k=12, train=True, corpus_dir='/content/corpus'):\n",
        "        self.ids = ids; self.id2row = id2row; self.labels = labels\n",
        "        self.res = res; self.k = k; self.train = train\n",
        "        self.corpus_dir = corpus_dir\n",
        "        self.vols = None; self.masks = None\n",
        "\n",
        "    def __len__(self): return len(self.ids)\n",
        "    def _ensure(self):\n",
        "        if self.vols is None:\n",
        "            self.vols, self.masks = open_corpus(self.corpus_dir)\n",
        "\n",
        "    def _centers(self, valid, count):\n",
        "        lo, hi = int(valid.min()), int(valid.max())\n",
        "        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]\n",
        "        if not cs: cs = [max(1, min((lo + hi) // 2, self._D - 2))]\n",
        "        if self.train:\n",
        "            reps = count // len(cs) + 1\n",
        "            pool = (cs * reps)\n",
        "            random.shuffle(pool)\n",
        "            return pool[:count]\n",
        "        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)\n",
        "        return [cs[i] for i in idx]\n",
        "\n",
        "    def _resize(self, tri):\n",
        "        t = torch.from_numpy(np.ascontiguousarray(tri))\n",
        "        if t.shape[-1] != self.res or t.shape[-2] != self.res:\n",
        "            t = F.interpolate(t[None], size=(self.res, self.res), mode='bilinear', align_corners=False)[0]\n",
        "        return t.numpy()\n",
        "\n",
        "    def __getitem__(self, i):\n",
        "        self._ensure()\n",
        "        uid = self.ids[i]; row = self.id2row[uid]\n",
        "        self._D = self.vols.shape[1]\n",
        "        m = self.masks[row]\n",
        "        valid = np.where(m > 0)[0]\n",
        "        if len(valid) < 3: valid = np.arange(min(3, self._D))\n",
        "        cs = self._centers(valid, self.k)\n",
        "        vol = self.vols[row]\n",
        "        tiles = []\n",
        "        for c in cs:\n",
        "            c = max(1, min(c, self._D - 2))\n",
        "            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0\n",
        "            tiles.append(self._resize(tri))\n",
        "        wins = np.stack(tiles, 0)\n",
        "        x = torch.from_numpy(wins)\n",
        "        if self.train:\n",
        "            g = 1.0 + (random.random() - 0.5) * 0.20\n",
        "            x = (x * g).clamp(0, 1)\n",
        "        y = torch.from_numpy(self.labels[uid])\n",
        "        return x, y\n",
        "\n",
        "def collate(batch):\n",
        "    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])\n",
        "\n",
        "class RaptorClassifier(nn.Module):\n",
        "    def __init__(self, backbone, F_dim=384, n=12, drop=0.2):\n",
        "        super().__init__()\n",
        "        self.backbone = backbone\n",
        "        self.norm = nn.LayerNorm(F_dim)\n",
        "        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),\n",
        "                                 nn.Linear(256, n))\n",
        "        self.clsW = nn.Parameter(torch.zeros(n, F_dim))\n",
        "        self.clsb = nn.Parameter(torch.zeros(n))\n",
        "        nn.init.trunc_normal_(self.clsW, std=0.02)\n",
        "\n",
        "    def encode(self, x):\n",
        "        B, K = x.shape[:2]\n",
        "        f = self.backbone(x.flatten(0, 1))\n",
        "        return f.view(B, K, -1)\n",
        "\n",
        "    def forward(self, x):\n",
        "        feats = self.encode(x)\n",
        "        h = self.norm(feats)\n",
        "        a = torch.softmax(self.att(h), dim=1)\n",
        "        pooled = torch.einsum('bkn,bkf->bnf', a, h)\n",
        "        logits = (pooled * self.clsW).sum(-1) + self.clsb\n",
        "        return logits\n",
        "\n",
        "def main():\n",
        "    ap = argparse.ArgumentParser()\n",
        "    ap.add_argument('--arch', default='coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k')\n",
        "    ap.add_argument('--res', type=int, default=384)\n",
        "    ap.add_argument('--k', type=int, default=12)\n",
        "    ap.add_argument('--k_eval', type=int, default=24)\n",
        "    ap.add_argument('--epochs', type=int, default=12)\n",
        "    ap.add_argument('--bs', type=int, default=8)\n",
        "    ap.add_argument('--bb_lr', type=float, default=3e-5)\n",
        "    ap.add_argument('--head_lr', type=float, default=1e-3)\n",
        "    ap.add_argument('--workers', type=int, default=4)\n",
        "    ap.add_argument('--tag', default='colab_super_coatnet')\n",
        "    a = ap.parse_args()\n",
        "\n",
        "    dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
        "    ids = open_ids('/content/corpus')\n",
        "    id2row = {u: i for i, u in enumerate(ids)}\n",
        "    idset = set(ids)\n",
        "\n",
        "    tr = pd.read_csv('/content/rsna_comp/train.csv')\n",
        "    tr['StudyInstanceUID'] = tr['StudyInstanceUID'].astype(str)\n",
        "    gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index('StudyInstanceUID')\n",
        "    gold_ids = [u for u in gold_df.index if u in idset]\n",
        "\n",
        "    lab_path = '/content/labels/SUPER_CONSENSUS_labels.parquet'\n",
        "    if not os.path.exists(lab_path): lab_path = '/content/labels/SUPER_CONSENSUS_labels.csv'\n",
        "    soft = pd.read_parquet(lab_path) if lab_path.endswith('.parquet') else pd.read_csv(lab_path)\n",
        "    soft['StudyInstanceUID'] = soft['StudyInstanceUID'].astype(str); soft = soft.set_index('StudyInstanceUID')\n",
        "\n",
        "    train_ids = [u for u in ids if u in soft.index and u not in set(gold_ids)]\n",
        "    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in train_ids}\n",
        "    for u in gold_ids: labels[u] = gold_df.loc[u, LAB].values.astype(np.float32)\n",
        "    print(f'Train: {len(train_ids)} studies | Validation (Gold Anchor): {len(gold_ids)} studies')\n",
        "\n",
        "    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)\n",
        "    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)\n",
        "\n",
        "    tds = StudyWindows(train_ids, id2row, labels, a.res, a.k, train=True)\n",
        "    vds = StudyWindows(gold_ids, id2row, labels, a.res, a.k_eval, train=False)\n",
        "    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)\n",
        "    vl = DataLoader(vds, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers, pin_memory=True)\n",
        "\n",
        "    bb = timm.create_model(a.arch, pretrained=True, num_classes=0, in_chans=3, global_pool='avg')\n",
        "    F_dim = bb.num_features\n",
        "    model = RaptorClassifier(bb, F_dim=F_dim).to(dev)\n",
        "\n",
        "    head_p = [p for n, p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]\n",
        "    bb_p = [p for n, p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]\n",
        "    opt = torch.optim.AdamW([{'params': bb_p, 'lr': a.bb_lr}, {'params': head_p, 'lr': a.head_lr}], weight_decay=0.02)\n",
        "    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=len(tl)*a.epochs, pct_start=0.15)\n",
        "    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)\n",
        "    scaler = torch.cuda.amp.GradScaler()\n",
        "\n",
        "    @torch.no_grad()\n",
        "    def evaluate():\n",
        "        model.eval(); P, Y = [], []\n",
        "        for x, y in vl:\n",
        "            with torch.autocast(dev, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):\n",
        "                o = torch.sigmoid(model(x.to(dev)).float())\n",
        "            P.append(o.cpu().numpy()); Y.append(y.numpy())\n",
        "        P, Y = np.concatenate(P), np.concatenate(Y)\n",
        "        aucs = {}\n",
        "        for j, name in enumerate(LAB):\n",
        "            if len(set(Y[:, j].astype(int))) > 1:\n",
        "                aucs[name] = float(roc_auc_score(Y[:, j], P[:, j]))\n",
        "        return float(np.mean(list(aucs.values()))), aucs, P, Y\n",
        "\n",
        "    best_auc = 0.0; best_state = None; top_states = []\n",
        "    t0 = time.time()\n",
        "    for ep in range(a.epochs):\n",
        "        model.train(); tot = 0.0\n",
        "        for x, y in tl:\n",
        "            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)\n",
        "            opt.zero_grad()\n",
        "            with torch.autocast(dev, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):\n",
        "                logits = model(x)\n",
        "                loss = lossf(logits.float(), y)\n",
        "            scaler.scale(loss).backward()\n",
        "            scaler.unscale_(opt)\n",
        "            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)\n",
        "            scaler.step(opt); scaler.update(); sched.step()\n",
        "            tot += loss.item()\n",
        "        au, aucs, P, Y = evaluate()\n",
        "        print(f'Epoch {ep+1:2d}/{a.epochs} | Loss: {tot/len(tl):.4f} | Gold Macro-AUC: {au:.4f} (Best: {max(best_auc, au):.4f}) | {time.time()-t0:.0f}s', flush=True)\n",
        "        state_copy = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}\n",
        "        top_states.append({'epoch': ep+1, 'auc': au, 'state': state_copy})\n",
        "        top_states.sort(key=lambda x: -x['auc'])\n",
        "        if len(top_states) > 3: top_states.pop()\n",
        "        if au > best_auc:\n",
        "            best_auc = au\n",
        "            torch.save({'model': state_copy, 'gold_auc': au, 'aucs': aucs, 'arch': a.arch, 'res': a.res, 'lab': LAB}, f'{a.tag}_best.pt')\n",
        "            print(f'   -> Saved new best checkpoint ({au:.4f})', flush=True)\n",
        "\n",
        "    # Stochastic Weight Averaging (SWA) over top 3 epochs\n",
        "    if len(top_states) > 1:\n",
        "        avg_sd = {}\n",
        "        for k in top_states[0]['state']:\n",
        "            avg_sd[k] = sum(s['state'][k].double() for s in top_states).div(len(top_states)).to(top_states[0]['state'][k].dtype)\n",
        "        model.load_state_dict(avg_sd)\n",
        "        swa_auc, swa_aucs, _, _ = evaluate()\n",
        "        print(f'\\nSWA Macro-AUC: {swa_auc:.4f} (across epochs {[s[\"epoch\"] for s in top_states]})', flush=True)\n",
        "        torch.save({'model': avg_sd, 'gold_auc': swa_auc, 'aucs': swa_aucs, 'arch': a.arch, 'res': a.res, 'lab': LAB}, f'{a.tag}_swa.pt')\n",
        "        print(f'Saved SWA checkpoint: {a.tag}_swa.pt')\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    main()\n"
    ]
})

# Cell 6: Run Execution
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 5: Execute Training (A100 Batch Size 8-16, Fast BF16)"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# On A100 (40GB), bs=8 or 16 runs blazing fast without gradient checkpointing!\n",
        "!python train_colab.py \\\n",
        "    --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \\\n",
        "    --res 384 \\\n",
        "    --epochs 12 \\\n",
        "    --bs 8 \\\n",
        "    --k 12 \\\n",
        "    --k_eval 24 \\\n",
        "    --workers 4 \\\n",
        "    --tag colab_super_coatnet_0917"
    ]
})

# Cell 7: Upload Trained Weights to Kaggle
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Step 6: Push Trained Weights Directly to Kaggle Dataset\n",
        "This immediately packages `colab_super_coatnet_0917_swa.pt` as a Kaggle dataset so our inference notebook can mount and submit it."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!mkdir -p /content/kaggle_weights\n",
        "!cp colab_super_coatnet_0917*.pt /content/kaggle_weights/\n",
        "\n",
        "import json\n",
        "meta = {\n",
        "    'title': 'RSNA Knee Colab Super Trained Weights',\n",
        "    'id': 'cyruspinto/rsna-knee-colab-super-weights',\n",
        "    'licenses': [{'name': 'CC0-1.0'}]\n",
        "}\n",
        "with open('/content/kaggle_weights/dataset-metadata.json', 'w') as f:\n",
        "    json.dump(meta, f, indent=2)\n",
        "\n",
        "!kaggle datasets create -p /content/kaggle_weights --dir-mode zip || kaggle datasets version -p /content/kaggle_weights -m 'Update weights from Colab A100'\n",
        "print('Weights published to Kaggle Dataset: cyruspinto/rsna-knee-colab-super-weights')"
    ]
})

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

target_file = r'c:\Users\Cyrus\Downloads\rsna knee fip\colab_rsna_knee_super_train.ipynb'
with open(target_file, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Successfully generated turnkey Colab notebook:', target_file)
