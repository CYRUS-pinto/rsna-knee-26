"""
Candidate V22: Full Quint-Engine Master Ensemble Notebook Assembler.
Injects:
1. Engine 6 architecture (ConvNeXt-Small + 2-layer BiLSTM + SequenceGatedAttentionPooling)
2. Engine 6 inference script (/tmp/engine6_run.py)
3. Stage 3 Quint-Engine Ensembling:
   - Raptor 384 + Raptor 224 (30%)
   - ConvNeXt R6 3D Joint Crops (30%)
   - Engine 5 Multi-Plane Fluid Specialist (20%)
   - Engine 6 Native Joint Sequence Specialist (20%)
"""

import os
import sys
import json
import shutil
from pathlib import Path

def assemble_v22():
    v21_path = Path('submissions/v21_quad_engine_specialist/rsna-knee-v21-quad-engine-blend.ipynb')
    if not v21_path.exists():
        raise FileNotFoundError(f"Missing {v21_path}")

    with open(v21_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Cell 0: Header
    nb['cells'][0]['source'] = [
        "# RSNA Knee Abnormality Detection: Candidate V22 Quint-Engine Master Blend\n",
        "# 5 Orthogonal Anatomical Engines for 0.965+ Podium Contention:\n",
        "# 1. Raptor-384 Whole Volume (CoAtNet-d4 SWA)\n",
        "# 2. Raptor-224 Whole Volume (Multi-scale center crop)\n",
        "# 3. ConvNeXt R6 3D Joint Crops (SKM-TEA AnchorBank + Flip TTA)\n",
        "# 4. Engine 5 2.5D Multi-Plane Fluid Specialist (Axial Fluid / Synovitis 0.8917 AUC)\n",
        "# 5. Engine 6 Native 2.5D Joint Sequence MIL (ConvNeXt-Small + BiLSTM on Sagittal/Coronal)\n"
    ]

    # Engine 6 Inference Script (Cell 9b to be inserted before Stage 3)
    engine6_script_source = [
        "%%writefile /tmp/engine6_run.py\n",
        "import os\n",
        "import sys\n",
        "import time\n",
        "import glob\n",
        "from pathlib import Path\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import timm\n",
        "\n",
        "ALL_LABELS = [\n",
        "    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',\n",
        "    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',\n",
        "    'Synovitis', \"Baker's\", 'Contusion', 'Fracture'\n",
        "]\n",
        "\n",
        "SPECIALIST_TARGETS = {\n",
        "    'sagittal': ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Contusion'],\n",
        "    'coronal': ['MCL', 'Medial OA', 'Lateral OA']\n",
        "}\n",
        "\n",
        "PLANE_SLOTS = {\n",
        "    'sagittal': [(0, 26, 8), (26, 48, 8)],     # 16 sequential slices\n",
        "    'coronal': [(48, 66, 8), (66, 78, 8)]      # 16 sequential slices\n",
        "}\n",
        "\n",
        "class SequenceGatedAttentionPooling(nn.Module):\n",
        "    def __init__(self, in_features, num_classes, hidden_dim=256, dropout=0.1):\n",
        "        super().__init__()\n",
        "        self.num_classes = num_classes\n",
        "        self.norm = nn.LayerNorm(in_features)\n",
        "        self.attn_v = nn.ModuleList([nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh(), nn.Dropout(dropout)) for _ in range(num_classes)])\n",
        "        self.attn_u = nn.ModuleList([nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout)) for _ in range(num_classes)])\n",
        "        self.attn_w = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_classes)])\n",
        "        self.classifiers = nn.ModuleList([nn.Linear(in_features, 1) for _ in range(num_classes)])\n",
        "    def forward(self, seq_feats):\n",
        "        B, K, D = seq_feats.shape\n",
        "        h = self.norm(seq_feats)\n",
        "        logits_list = []\n",
        "        for c in range(self.num_classes):\n",
        "            v = self.attn_v[c](h)\n",
        "            u = self.attn_u[c](h)\n",
        "            a = self.attn_w[c](v * u).squeeze(-1)\n",
        "            alpha = F.softmax(a, dim=1).unsqueeze(-1)\n",
        "            pooled = torch.sum(h * alpha, dim=1)\n",
        "            logits_list.append(self.classifiers[c](pooled))\n",
        "        return torch.cat(logits_list, dim=-1)\n",
        "\n",
        "class Engine6SpecialistSequenceMIL(nn.Module):\n",
        "    def __init__(self, num_classes, backbone_name='convnext_small', pretrained=False, lstm_hidden=256):\n",
        "        super().__init__()\n",
        "        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, in_chans=3, num_classes=0)\n",
        "        feat_dim = self.backbone.num_features\n",
        "        self.bilstm = nn.LSTM(input_size=feat_dim, hidden_size=lstm_hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)\n",
        "        self.pool = SequenceGatedAttentionPooling(in_features=lstm_hidden * 2, num_classes=num_classes)\n",
        "    def forward(self, x):\n",
        "        B, K, C, H, W = x.shape\n",
        "        x_flat = x.view(B * K, C, H, W)\n",
        "        feats = self.backbone(x_flat).view(B, K, -1)\n",
        "        lstm_out, _ = self.bilstm(feats)\n",
        "        return self.pool(lstm_out)\n",
        "\n",
        "def _find(*names, roots=['/kaggle/input', '.']):\n",
        "    for r in roots:\n",
        "        if os.path.exists(r):\n",
        "            for root, _, files in os.walk(r):\n",
        "                for n in names:\n",
        "                    if n in files:\n",
        "                        return os.path.join(root, n)\n",
        "    return None\n",
        "\n",
        "def run_engine6_inference():\n",
        "    print('--- Running Engine 6 Native Joint-Space Sequence Inference ---')\n",
        "    t0 = time.time()\n",
        "    device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
        "    test_csv = _find('test.csv')\n",
        "    if not test_csv:\n",
        "        print('Missing test.csv, generating dummy predictions')\n",
        "        return\n",
        "    df_test = pd.read_csv(test_csv, dtype={'StudyInstanceUID': str})\n",
        "    test_ids = df_test['StudyInstanceUID'].tolist()\n",
        "    out_df = pd.DataFrame({'StudyInstanceUID': test_ids})\n",
        "    for col in ALL_LABELS:\n",
        "        out_df[col] = 0.5\n",
        "\n",
        "    # Look for Engine 6 specialist weights\n",
        "    sag_w = _find('engine6_sagittal_sequence_best.pth', roots=['/kaggle/input', '.', 'weights'])\n",
        "    cor_w = _find('engine6_coronal_sequence_best.pth', roots=['/kaggle/input', '.', 'weights'])\n",
        "    print(f'Engine 6 Weights: Sagittal={sag_w}, Coronal={cor_w}')\n",
        "\n",
        "    vol_dir = Path('/kaggle/working/e4_vols')\n",
        "    if not vol_dir.exists():\n",
        "        vol_dir = Path('/kaggle/working/our_runs')\n",
        "\n",
        "    # 1. Sagittal Specialist\n",
        "    if sag_w and os.path.exists(sag_w):\n",
        "        targets = SPECIALIST_TARGETS['sagittal']\n",
        "        model = Engine6SpecialistSequenceMIL(num_classes=len(targets), backbone_name='convnext_small', pretrained=False).to(device)\n",
        "        model.load_state_dict(torch.load(sag_w, map_location=device), strict=True)\n",
        "        model.eval()\n",
        "        print(f'Loaded Engine 6 Sagittal Specialist from {sag_w}')\n",
        "        for uid in test_ids:\n",
        "            fpath = vol_dir / f'{uid}.npy'\n",
        "            if fpath.exists():\n",
        "                vol = np.load(fpath, mmap_mode='r')\n",
        "                slices = []\n",
        "                for start, end, step in PLANE_SLOTS['sagittal']:\n",
        "                    idx = np.linspace(start, end - 1, step, dtype=int)\n",
        "                    for i in idx:\n",
        "                        c_prev = max(0, i - 1)\n",
        "                        c_next = min(vol.shape[0] - 1, i + 1)\n",
        "                        slab = np.stack([vol[c_prev], vol[i], vol[c_next]], axis=0)\n",
        "                        slices.append(slab)\n",
        "                inp = torch.from_numpy(np.stack(slices[:16], axis=0)).unsqueeze(0).float().to(device)\n",
        "                with torch.no_grad():\n",
        "                    probs = torch.sigmoid(model(inp)).cpu().numpy()[0]\n",
        "                for col_idx, col in enumerate(targets):\n",
        "                    out_df.loc[out_df['StudyInstanceUID'] == uid, col] = float(probs[col_idx])\n",
        "        del model\n",
        "        torch.cuda.empty_cache()\n",
        "\n",
        "    # 2. Coronal Specialist\n",
        "    if cor_w and os.path.exists(cor_w):\n",
        "        targets = SPECIALIST_TARGETS['coronal']\n",
        "        model = Engine6SpecialistSequenceMIL(num_classes=len(targets), backbone_name='convnext_small', pretrained=False).to(device)\n",
        "        model.load_state_dict(torch.load(cor_w, map_location=device), strict=True)\n",
        "        model.eval()\n",
        "        print(f'Loaded Engine 6 Coronal Specialist from {cor_w}')\n",
        "        for uid in test_ids:\n",
        "            fpath = vol_dir / f'{uid}.npy'\n",
        "            if fpath.exists():\n",
        "                vol = np.load(fpath, mmap_mode='r')\n",
        "                slices = []\n",
        "                for start, end, step in PLANE_SLOTS['coronal']:\n",
        "                    idx = np.linspace(start, end - 1, step, dtype=int)\n",
        "                    for i in idx:\n",
        "                        c_prev = max(0, i - 1)\n",
        "                        c_next = min(vol.shape[0] - 1, i + 1)\n",
        "                        slab = np.stack([vol[c_prev], vol[i], vol[c_next]], axis=0)\n",
        "                        slices.append(slab)\n",
        "                inp = torch.from_numpy(np.stack(slices[:16], axis=0)).unsqueeze(0).float().to(device)\n",
        "                with torch.no_grad():\n",
        "                    probs = torch.sigmoid(model(inp)).cpu().numpy()[0]\n",
        "                for col_idx, col in enumerate(targets):\n",
        "                    out_df.loc[out_df['StudyInstanceUID'] == uid, col] = float(probs[col_idx])\n",
        "        del model\n",
        "        torch.cuda.empty_cache()\n",
        "\n",
        "    out_path = Path('/kaggle/working/engine6_sub.csv')\n",
        "    out_df.to_csv(out_path, index=False)\n",
        "    print(f'Engine 6 Inference complete in {time.time() - t0:.1f}s -> Saved {out_path}')\n",
        "\n",
        "if __name__ == '__main__':\n",
        "    run_engine6_inference()\n"
    ]

    # Execution cell for Engine 6
    engine6_exec_source = [
        "# Stage 2c Execution: Engine 6 Native Joint Sequence Specialists\n",
        "!python3 /tmp/engine6_run.py\n"
    ]

    # Blend cell (Cell 10 modification to Quint-Engine)
    blend_source = [
        "# Stage 3: Candidate V22 Quint-Engine Master Specialist Blend\n",
        "# 5 Orthogonal Engines:\n",
        "# 1. Raptor-384 Whole Volume (CoAtNet-d4)\n",
        "# 2. Raptor-224 Whole Volume (Multi-scale)\n",
        "# 3. ConvNeXt R6 3D Joint Crops (AnchorBank)\n",
        "# 4. Engine 5 2.5D Multi-Plane Fluid Specialist (Axial Fluid / Synovitis)\n",
        "# 5. Engine 6 Native 2.5D Joint Sequence MIL (Sagittal/Coronal)\n",
        "import csv\n",
        "import json\n",
        "import time\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from pathlib import Path\n",
        "\n",
        "DATA = Path('/kaggle/input/rsna-knee-abnormality-detection')\n",
        "WORK = Path('/kaggle/working')\n",
        "pred = Path('/kaggle/working/our_runs')\n",
        "\n",
        "W_RAPTOR, W_OURS = 0.5, 0.5\n",
        "A224 = 0.35\n",
        "assert 0.0 < A224 < 1.0\n",
        "\n",
        "test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()\n",
        "ss = DATA / 'sample_submission.csv'\n",
        "ours = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})\n",
        "e5 = pd.read_csv(WORK / 'engine5_sub.csv', dtype={'StudyInstanceUID': str})\n",
        "e6 = pd.read_csv(WORK / 'engine6_sub.csv', dtype={'StudyInstanceUID': str})\n",
        "cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else list(ours.columns)\n",
        "labels = cols[1:]\n",
        "\n",
        "raw = pd.read_csv(WORK / 'raptor_raw.csv', dtype={'StudyInstanceUID': str})\n",
        "rsub = pd.read_csv(WORK / 'raptor_sub.csv', dtype={'StudyInstanceUID': str})\n",
        "raw224 = pd.read_csv(WORK / 'raptor224_raw.csv', dtype={'StudyInstanceUID': str})\n",
        "rsub224 = pd.read_csv(WORK / 'raptor224_sub.csv', dtype={'StudyInstanceUID': str})\n",
        "\n",
        "ours = ours.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "e5 = e5.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "e6 = e6.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "raw = raw.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "rsub = rsub.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "raw224 = raw224.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "rsub224 = rsub224.set_index('StudyInstanceUID').loc[test_ids, labels]\n",
        "n = len(test_ids)\n",
        "\n",
        "r384_fail = (raw.values == 0.5).all(axis=1)\n",
        "r224_fail = (raw224.values == 0.5).all(axis=1)\n",
        "rap_fail = r384_fail & r224_fail\n",
        "\n",
        "scored = set()\n",
        "for f in pred.glob('worker*/predictions.csv'):\n",
        "    with open(f, encoding='utf-8') as fh:\n",
        "        for r in csv.DictReader(fh):\n",
        "            try:\n",
        "                v = [float(r[L]) for L in labels]\n",
        "            except (KeyError, ValueError, TypeError):\n",
        "                continue\n",
        "            if all(0. <= x <= 1. for x in v):\n",
        "                scored.add(r['StudyInstanceUID'])\n",
        "our_fail = np.array([u not in scored for u in test_ids])\n",
        "\n",
        "def pct(v):\n",
        "    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values\n",
        "\n",
        "def pct_on(v, ok):\n",
        "    r = np.full(len(v), np.nan)\n",
        "    if ok.any():\n",
        "        r[ok] = pct(np.asarray(v, dtype=np.float64)[ok])\n",
        "    return r\n",
        "\n",
        "out = np.zeros((n, len(labels)))\n",
        "\n",
        "# Focal findings apply generalized power-mean (p=1.4)\n",
        "POWER_EXPONENTS = {\n",
        "    'ACL': 1.4, 'Medial Meniscus': 1.4, 'Lateral Meniscus': 1.4, 'Fracture': 1.4,\n",
        "    'MCL': 1.0, 'Medial OA': 1.0, 'Lateral OA': 1.0, 'PF OA': 1.0,\n",
        "    'Effusion': 1.0, 'Synovitis': 1.0, \"Baker's\": 1.0, 'Contusion': 1.0\n",
        "}\n",
        "\n",
        "for j, c in enumerate(labels):\n",
        "    r384, r224 = pct_on(raw[c].values, ~r384_fail), pct_on(raw224[c].values, ~r224_fail)\n",
        "    rr = np.where(r384_fail, r224, np.where(r224_fail, r384, (1.0 - A224) * r384 + A224 * r224))\n",
        "    ro = pct_on(ours[c].values, ~our_fail)\n",
        "    re5 = pct(e5[c].values)\n",
        "    re6 = pct(e6[c].values)\n",
        "    \n",
        "    # Weights depend on specialist domain\n",
        "    if c in ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Contusion']:\n",
        "        # Sagittal Sequence Specialist is primary\n",
        "        w_rr, w_ro, w_e5, w_e6 = 0.30, 0.30, 0.10, 0.30\n",
        "    elif c in ['MCL', 'Medial OA', 'Lateral OA']:\n",
        "        # Coronal Sequence Specialist is primary\n",
        "        w_rr, w_ro, w_e5, w_e6 = 0.30, 0.30, 0.10, 0.30\n",
        "    elif c in ['Synovitis', 'Effusion', 'PF OA']:\n",
        "        # Axial Fluid Specialist is primary\n",
        "        w_rr, w_ro, w_e5, w_e6 = 0.35, 0.35, 0.25, 0.05\n",
        "    else:\n",
        "        w_rr, w_ro, w_e5, w_e6 = 0.35, 0.35, 0.15, 0.15\n",
        "        \n",
        "    p = POWER_EXPONENTS.get(c, 1.0)\n",
        "    if p != 1.0:\n",
        "        rr_c = np.clip(rr, 1e-6, 1.0)\n",
        "        ro_c = np.clip(ro, 1e-6, 1.0)\n",
        "        re5_c = np.clip(re5, 1e-6, 1.0)\n",
        "        re6_c = np.clip(re6, 1e-6, 1.0)\n",
        "        b = (w_rr * (rr_c ** p) + w_ro * (ro_c ** p) + w_e5 * (re5_c ** p) + w_e6 * (re6_c ** p)) ** (1.0 / p)\n",
        "    else:\n",
        "        b = w_rr * rr + w_ro * ro + w_e5 * re5 + w_e6 * re6\n",
        "        \n",
        "    b[rap_fail] = ro[rap_fail]\n",
        "    b[our_fail] = rr[our_fail]\n",
        "    b[rap_fail & our_fail] = 0.5\n",
        "    out[:, j] = b\n",
        "\n",
        "final_sub = pd.DataFrame(out, index=test_ids, columns=labels).reset_index()\n",
        "final_sub.rename(columns={'index': 'StudyInstanceUID'}, inplace=True)\n",
        "final_sub.to_csv(WORK / 'submission.csv', index=False)\n",
        "print(f'Candidate V22 Quint-Engine submission.csv successfully generated ({len(final_sub)} rows)!')\n"
    ]

    # Create new notebook cells list
    new_cells = []
    for i, cell in enumerate(nb['cells']):
        if i == 10:
            # Insert Engine 6 cells before the blend cell
            new_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": engine6_script_source
            })
            new_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": engine6_exec_source
            })
            # Insert the new blend cell
            new_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": blend_source
            })
        else:
            new_cells.append(cell)

    nb['cells'] = new_cells

    out_dir = Path('submissions/v22_quint_engine_master')
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "id": "cyruspinto/rsna-knee-v22-quint-engine-blend",
        "title": "RSNA Knee V22 Quint Engine Blend",
        "code_file": "rsna-knee-v22-quint-engine-blend.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "heliosli/rsna-knee-r6-frozen-fe055406-w1",
            "nartaa/rsna-knee-publication-swa-weights-20261007",
            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
            "cyruspinto/rsna-knee-engine5-weights",
            "cyruspinto/rsna-knee-engine6-weights"
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "machine_shape": "NvidiaTeslaT4"
    }
    with open(out_dir / 'kernel-metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    out_nb = out_dir / 'rsna-knee-v22-quint-engine-blend.ipynb'
    with open(out_nb, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully assembled complete Candidate V22 in {out_nb} ({len(nb['cells'])} cells)")

if __name__ == '__main__':
    assemble_v22()
