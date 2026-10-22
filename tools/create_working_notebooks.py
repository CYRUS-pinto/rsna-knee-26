import json
from pathlib import Path

# Read the base notebooks
engine12_src = Path(r'C:\Users\Cyrus\Downloads\colab_engine12_fold1_master.ipynb')
engine13_src = Path(r'C:\Users\Cyrus\Downloads\colab_engine13_fold1_master.ipynb')

with open(r'C:\Users\Cyrus\Downloads\colab_engine12_fold1_master.ipynb', 'r', encoding='utf-8') as f:
    engine12_base = json.load(f)

with open(r'C:\Users\Cyrus\Downloads\colab_engine13_fold1_master.ipynb', 'r', encoding='utf-8') as f:
    engine13_base = json.load(f)

# Build Engine 12 multi-fold source
engine12_multi = '''# Multi-Fold Loop: Folds 2, 3, 4
for FOLD in [2, 3, 4]:
    print(f"=== Starting Fold {FOLD} ===", flush=True)
    
    # Write training script for this fold
    script = (
        "RSNA Knee Engine 12 - ConvNeXt-Large-384 Gated MIL (Fold {FOLD} of 5)\n"
        "5-Fold Stratified Cross-Validation + Official 58 Gold Radiologist Anchoring\n"
        "\\n"
        "import os, sys, time, math, random\n"
        "import numpy as np, pandas as pd\n"
        "import torch, torch.nn as nn, torch.nn.functional as F\n"
        "from torch.utils.data import Dataset, DataLoader\n"
        "from sklearn.metrics import roc_auc_score\n"
        "import timm\n\n"
        "TARGETS = [\n"
        "    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',\n"
        "    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',\n"
        "    'Synovitis', \"Baker's\", 'Contusion', 'Fracture'\n"
        "]\n\n"
        "class TwoPartVols:\\n"
        "    def __init__(self, a, b):\\n"
        "        self.a, self.b = a, b\\n"
        "        self.n_a = a.shape[0]\\n"
        "        self.shape = (a.shape[0] + (b.shape[0] if b is not None else 0),) + tuple(a.shape[1:])\\n"
        "    def __len__(self): return self.shape[0]\\n"
        "    def __getitem__(self, row):\\n"
        "        return self.a[row] if row < self.n_a else self.b[row - self.n_a]\\n"
        "\\n"
        "class StudyWindowsDataset(Dataset):\\n"
        "    def __init__(self, ids, id2row, labels, vols, masks, res=384, k=16, train=False):\\n"
        "        self.ids = ids\\n"
        "        self.id2row = id2row\\n"
        "        self.labels = labels\\n"
        "        self.vols = vols\\n"
        "        self.masks = masks\\n"
        "        self.res = res\\n"
        "        self.k = k\\n"
        "        self.train = train\\n"
        "\\n"
        "    def __len__(self): return len(self.ids)\\n"
        "\\n"
        "    def _centers(self, valid, count, d_len):\\n"
        "        lo, hi = int(valid.min()), int(valid.max())\\n"
        "        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]\\n"
        "        if not cs:\\n"
        "            cs = [max(1, min((lo + hi) // 2, d_len - 2))]\\n"
        "        if self.train:\\n"
        "            reps = count // len(cs) + 1\\n"
        "            pool = (cs * reps)\\n"
        "            random.shuffle(pool)\\n"
        "            return sorted(pool[:count])\\n"
        "        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)\\n"
        "        return sorted([cs[i] for i in idx])\\n"
        "\\n"
        "    def __getitem__(self, i):\\n"
        "        uid = self.ids[i]\\n"
        "        row = self.id2row[uid]\\n"
        "        vol = self.vols[row]\\n"
        "        mask = self.masks[row] if self.masks is not None else None\\n"
        "        \\n"
        "        if mask is not None:\\n"
        "            valid = np.where(mask)[0]\\n"
        "        else:\\n"
        "            valid = np.arange(vol.shape[0])\\n"
        "\\n"
        "        centers = self._centers(valid, self.k, vol.shape[0])\\n"
        "\\n"
        "        windows = []\\n"
        "        for c in centers:\\n"
        "            if c - 1 >= 0 and c + 1 < vol.shape[0]:\\n"
        "                window = vol[c-1:c+2]  # 3 slices\\n"
        "            else:\\n"
        "                window = np.zeros((3, vol.shape[1], vol.shape[2]), dtype=vol.dtype)\\n"
        "                mid = min(max(c, 1), vol.shape[0] - 2)\\n"
        "                window = vol[mid-1:mid+2]\\n"
        "            # Resize to 384x384\\n"
        "            from skimage.transform import resize\\n"
        "            window = np.stack([resize(w, (self.res, self.res), order=1, preserve_range=True, anti_aliasing=True) for w in window])\\n"
        "            windows.append(window)\\n"
        "\\n"
        "        windows = np.stack(windows)  # (k, 3, H, W)\\n"
        "        windows = torch.from_numpy(windows).float() / 255.0\\n"
        "\\n"
        "        # Labels\\n"
        "        y = self.labels.loc[self.labels['StudyInstanceUID'] == uid, TARGETS].values\\n"
        "        if len(y) == 0:\\n"
        "            y = np.zeros(len(TARGETS), dtype=np.float32)\\n"
        "        else:\\n"
        "            y = y[0].astype(np.float32)\\n"
        "\\n"
        "        return windows, torch.from_numpy(y), uid\\n"
        "\\n"
        "\\n"
        "class ConvNeXtMILClassifier(nn.Module):\\n"
        "    def __init__(self, backbone, feat_dim=1536, num_targets=12, drop=0.2):\\n"
        "        super().__init__()\\n"
        "        self.backbone = backbone\\n"
        "        self.feat_dim = feat_dim\\n"
        "        self.norm = nn.LayerNorm(feat_dim)\\n"
        "        self.att = nn.Sequential(\\n"
        "            nn.Linear(feat_dim, 256),\\n"
        "            nn.Tanh(),\\n"
        "            nn.Dropout(drop),\\n"
        "            nn.Linear(256, num_targets)\\n"
        "        )\\n"
        "        self.clsW = nn.Parameter(torch.zeros(num_targets, feat_dim))\\n"
        "        self.clsb = nn.Parameter(torch.zeros(num_targets))\\n"
        "\\n"
        "    def encode(self, x):\\n"
        "        B, K, C, H, W = x.shape\\n"
        "        flat_x = x.view(B * K, C, H, W)\\n"
        "        chunk_size = 4\\n"
        "        feats = []\\n"
        "        for i in range(0, flat_x.size(0), chunk_size):\\n"
        "            f = self.backbone(flat_x[i:i + chunk_size])\\n"
        "            feats.append(f)\\n"
        "        feats = torch.cat(feats, dim=0)\\n"
        "        return feats.view(B, K, -1)\\n"
        "\\n"
        "    def forward(self, x):\\n"
        "        feats = self.encode(x)\\n"
        "        h = self.norm(feats)\\n"
        "        a = torch.softmax(self.att(h), dim=1)\\n"
        "        pooled = torch.einsum('bkn,bkf->bnf', a, h)\\n"
        "        logits = (pooled * self.clsW).sum(-1) + self.clsb\\n"
        "        return logits\\n"
        "\\n"
        "def run_engine12_inference():\\n"
        "    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')\\n"
        "    \\n"
        "    weight_paths = [\\n"
        "        Path(f'/content/weights/engine12_convnext_large_fold{FOLD}_best.pt'),\\n"
        "        Path(f'/content/weights/engine12_convnext_large_fold{FOLD}_best.pt'),\\n"
        "        Path('/kaggle/input/rsna-knee-engine12-convnext-large-weights/engine12_convnext_large_best.pt'),\\n"
        "    ]\\n"
        "    weight_file = next((p for p in weight_paths if p.is_file()), None)\\n"
        "\\n"
        "    if weight_file is None:\\n"
        "        print(f'[!] Engine 12 weights not found for Fold {FOLD}')\\n"
        "        return None\\n"
        "\\n"
        "    print(f'[+] Loading Engine 12 Fold {FOLD} from: {weight_file}', flush=True)\\n"
        "\\n"
        "    bb = timm.create_model('convnext_large.fb_in22k_ft_in1k_384', pretrained=False, num_classes=0)\\n"
        "    bb.set_grad_checkpointing(True)\\n"
        "    model = ConvNeXtMILClassifier(bb, feat_dim=1536, num_targets=12).to(device)\\n"
        "\\n"
        "    ckpt = torch.load(weight_file, map_location=device, weights_only=False)\\n"
        "    state_dict = ckpt['model'] if 'model' in ckpt else ckpt\\n"
        "    model.load_state_dict(state_dict, strict=False)\\n"
        "    model.eval()\\n"
        "\\n"
        "    val_dataset = StudyWindowsDataset(\\n"
        "        ids=list(id2row.keys()),\\n"
        "        id2row=id2row,\\n"
        "        labels=val_labels,\\n"
        "        vols=all_vols,\\n"
        "        masks=None,  # Will be loaded from corpus if available\\n"
        "        res=384,\\n"
        "        k=16,\\n"
        "        train=False\\n"
        "    )\\n"
        "\\n"
        "    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)\\n"
        "\\n"
        "    predictions = []\\n"
        "    uids = []\\n"
        "\\n"
        "    print(f'[*] Running Engine 12 Fold {FOLD} inference on {len(val_dataset)} validation studies...', flush=True)\\n"
        "    with torch.no_grad():\\n"
        "        for windows, _, uid in val_loader:\\n"
        "            windows = windows.to(device)\\n"
        "            logits = model(windows)\\n"
        "            probs = torch.sigmoid(logits).cpu().numpy()[0]\\n"
        "            predictions.append(probs)\\n"
        "            uids.append(uid)\\n"
        "\\n"
        "    pred_df = pd.DataFrame(predictions, columns=TARGETS)\\n"
        "    pred_df.insert(0, 'StudyInstanceUID', uids)\\n"
        "    pred_df.to_csv(f'/content/engine12_fold{FOLD}_oof.csv', index=False)\\n"
        "    print(f'[+] Engine 12 Fold {FOLD} OOF saved: {pred_df.shape}', flush=True)\\n"
        "    return pred_df\\n"
        "\\n"
        "    # Train this fold\\n"
        "    print(f'=== Starting Engine 12 Fold {FOLD} Training ===', flush=True)\\n"
        "    exec(open(f'train_engine12_fold{FOLD}.py').read())\\n"
        "\\n"
        "    # Publish weights\\n"
        "    import os, json, subprocess, shutil\\n"
        "    os.makedirs(f'/content/engine12_fold{FOLD}_upload', exist_ok=True)\\n"
        "    shutil.copy(f'/content/engine12_convnext_large_fold{FOLD}_best.pt', f'/content/engine12_fold{FOLD}_upload/')\\n"
        "\\n"
        "    meta = {\\n"
        "        'title': f'RSNA Knee Engine 12 ConvNeXt-Large Fold {FOLD} Weights',\\n"
        "        'id': f'cyruspinto/rsna-knee-engine12-fold{FOLD}-weights',\\n"
        "        'licenses': [{'name': 'CC0-1.0'}]\\n"
        "    }\\n"
        "    with open(f'/content/engine12_fold{FOLD}_upload/dataset-metadata.json', 'w') as f:\\n"
        "        json.dump(meta, f, indent=2)\\n"
        "\\n"
        "    subprocess.run(['kaggle', 'datasets', 'create', '-p', f'/content/engine12_fold{FOLD}_upload', '-r', 'zip'], capture_output=True)\\n"
        "    print(f'[+] Engine 12 Fold {FOLD} weights pushed to Kaggle', flush=True)\\n"
        "\\n"
        "    # Clear GPU memory for next fold\\n"
        "    import torch, gc\\n"
        "    torch.cuda.empty_cache()\\n"
        "    gc.collect()\\n"
        "    print(f'=== Fold {FOLD} Complete ===', flush=True)\\n"

print('=== All Engine 12 Folds Complete ===')
"""

# Build Engine 12 multi-fold notebook
def create_engine12_multi():
    with open(r'C:\Users\Cyrus\Downloads\colab_engine12_fold1_master.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Keep first 5 cells (env, auth, folds, labels, id2row)
    new_cells = engine12_base['cells'][:5]
    
    # Multi-fold loop cell
    new_cells.append({
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [engine12_multi]
    })
    
    # Final cell
    new_cells.append({
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': ['print("=== All Engine 12 Folds Complete ===")']
    })
    
    new_nb = {
        'cells': engine12_base['cells'][:5] + [
            {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [engine12_multi]},
            {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': ['print("=== All Engine 12 Folds Complete ===")']}
        ],
        'metadata': engine12_base['metadata'],
        'nbformat': engine12_base['nbformat'],
        'nbformat_minor': engine12_base['nbformat_minor']
    }
    
    with open(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb', 'w', encoding='utf-8') as f:
        json.dump({
            'cells': engine12_base['cells'][:5] + [
                {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [engine12_multi]},
                {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': ['print("=== All Engine 12 Folds Complete ===")']}
            ],
            'metadata': engine12_base['metadata'],
            'nbformat': engine12_base['nbformat'],
            'nbformat_minor': engine12_base['nbformat_minor']
        }, f, indent=1)
    print('Created Engine 12 multi-fold')

if __name__ == '__main__':
    import json
    from pathlib import Path
    
    # Read Engine 12 base
    with open(r'C:\Users\Cyrus\Downloads\colab_engine12_fold1_master.ipynb', 'r', encoding='utf-8') as f:
        engine12_base = json.load(f)
    
    # Engine 12 multi-fold source
    engine12_multi = '''# Multi-Fold Loop: Folds 2, 3, 4
for FOLD in [2, 3, 4]:
    print(f"=== Starting Fold {FOLD} ===", flush=True)
    
    # Write training script for this fold
    script = (
        "RSNA Knee Engine 12 - ConvNeXt-Large-384 Gated MIL (Fold {FOLD} of 5)\\n"
        "5-Fold Stratified Cross-Validation + Official 58 Gold Radiologist Anchoring\\n"
        "\\n"
        "import os, sys, time, math, random\\n"
        "import numpy as np, pandas as pd\\n"
        "import torch, torch.nn as nn, torch.nn.functional as F\\n"
        "from torch.utils.data import Dataset, DataLoader\\n"
        "from sklearn.metrics import roc_auc_score\\n"
        "import timm\\n\\n"
        "TARGETS = [\\n"
        "    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',\\n"
        "    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',\\n"
        "    'Synovitis', \"Baker's\", 'Contusion', 'Fracture'\\n"
        "]\\n\\n"
        "class TwoPartVols:\\n"
        "    def __init__(self, a, b):\\n"
        "        self.a, self.b = a, b\\n"
        "        self.n_a = a.shape[0]\\n"
        "        self.shape = (a.shape[0] + (b.shape[0] if b is not None else 0),) + tuple(a.shape[1:])\\n"
        "    def __len__(self): return self.shape[0]\\n"
        "    def __getitem__(self, row):\\n"
        "        return self.a[row] if row < self.n_a else self.b[row - self.n_a]\\n"
        "\\n"
        "class StudyWindowsDataset(Dataset):\\n"
        "    def __init__(self, ids, id2row, labels, vols, masks, res=384, k=16, train=False):\\n"
        "        self.ids = ids\\n"
        "        self.id2row = id2row\\n"
        "        self.labels = labels\\n"
        "        self.vols = vols\\n"
        "        self.masks = masks\\n"
        "        self.res = res\\n"
        "        self.k = k\\n"
        "        self.train = train\\n"
        "\\n"
        "    def __len__(self): return len(self.ids)\\n"
        "\\n"
        "    def _centers(self, valid, count, d_len):\\n"
        "        lo, hi = int(valid.min()), int(valid.max())\\n"
        "        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]\\n"
        "        if not cs:\\n"
        "            cs = [max(1, min((lo + hi) // 2, d_len - 2))]\\n"
        "        if self.train:\\n"
        "            reps = count // len(cs) + 1\\n"
        "            pool = (cs * reps)\\n"
        "            random.shuffle(pool)\\n"
        "            return sorted(pool[:count])\\n"
        "        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)\\n"
        "        return sorted([cs[i] for i in idx])\\n"
        "\\n"
        "    def __getitem__(self, i):\\n"
        "        uid = self.ids[i]\\n"
        "        row = self.id2row[uid]\\n"
        "        vol = self.vols[row]\\n"
        "        mask = self.masks[row] if self.masks is not None else None\\n"
        "\\n"
        "        if mask is not None:\\n"
        "            valid = np.where(mask)[0]\\n"
        "        else:\\n"
        "            valid = np.arange(vol.shape[0])\\n"
        "\\n"
        "        centers = self._centers(valid, self.k, vol.shape[0])\\n"
        "\\n"
        "        windows = []\\n"
        "        for c in centers:\\n"
        "            if c - 1 >= 0 and c + 1 < vol.shape[0]:\\n"
        "                window = vol[c-1:c+2]  # 3 slices\\n"
        "            else:\\n"
        "                window = np.zeros((3, vol.shape[1], vol.shape[2]), dtype=vol.dtype)\\n"
        "                mid = min(max(c, 1), vol.shape[0] - 2)\\n"
        "                window = vol[mid-1:mid+2]\\n"
        "            # Resize to 384x384\\n"
        "            from skimage.transform import resize\\n"
        "            window = np.stack([resize(w, (self.res, self.res), order=1, preserve_range=True, anti_aliasing=True) for w in window])\\n"
        "            windows.append(window)\\n"
        "\\n"
        "        windows = np.stack(windows)  # (k, 3, H, W)\\n"
        "        windows = torch.from_numpy(windows).float() / 255.0\\n"
        "\\n"
        "        # Labels\\n"
        "        y = self.labels.loc[self.labels['StudyInstanceUID'] == uid, TARGETS].values\\n"
        "        if len(y) == 0:\\n"
        "            y = np.zeros(len(TARGETS), dtype=np.float32)\\n"
        "        else:\\n"
        "            y = y[0].astype(np.float32)\\n"
        "\\n"
        "        return windows, torch.from_numpy(y), uid\\n"
        "\\n"
        "\\n"
        "class ConvNeXtMILClassifier(nn.Module):\\n"
        "    def __init__(self, backbone, feat_dim=1536, num_targets=12, drop=0.2):\\n"
        "        super().__init__()\\n"
        "        self.backbone = backbone\\n"
        "        self.feat_dim = feat_dim\\n"
        "        self.norm = nn.LayerNorm(feat_dim)\\n"
        "        self.att = nn.Sequential(\\n"
        "            nn.Linear(feat_dim, 256),\\n"
        "            nn.Tanh(),\\n"
        "            nn.Dropout(drop),\\n"
        "            nn.Linear(256, num_targets)\\n"
        "        )\\n"
        "        self.clsW = nn.Parameter(torch.zeros(num_targets, feat_dim))\\n"
        "        self.clsb = nn.Parameter(torch.zeros(num_targets))\\n"
        "\\n"
        "    def encode(self, x):\\n"
        "        B, K, C, H, W = x.shape\\n"
        "        flat_x = x.view(B * K, C, H, W)\\n"
        "        chunk_size = 4\\n"
        "        feats = []\\n"
        "        for i in range(0, flat_x.size(0), chunk_size):\\n"
        "            f = self.backbone(flat_x[i:i + chunk_size])\\n"
        "            feats.append(f)\\n"
        "        feats = torch.cat(feats, dim=0)\\n"
        "        return feats.view(B, K, -1)\\n"
        "\\n"
        "    def forward(self, x):\\n"
        "        feats = self.encode(x)\\n"
        "        h = self.norm(feats)\\n"
        "        a = torch.softmax(self.att(h), dim=1)\\n"
        "        pooled = torch.einsum('bkn,bkf->bnf', a, h)\\n"
        "        logits = (pooled * self.clsW).sum(-1) + self.clsb\\n"
        "        return logits\\n"
        "\\n"
        "def run_engine12_inference():\\n"
        "    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')\\n"
        "    \\n"
        "    weight_paths = [\\n"
        "        Path(f'/content/weights/engine12_convnext_large_fold{FOLD}_best.pt'),\\n"
        "        Path(f'/content/weights/engine12_convnext_large_fold{FOLD}_best.pt'),\\n"
        "        Path('/kaggle/input/rsna-knee-engine12-convnext-large-weights/engine12_convnext_large_best.pt'),\\n"
        "    ]\\n"
        "    weight_file = next((p for p in weight_paths if p.is_file()), None)\\n"
        "\\n"
        "    if weight_file is None:\\n"
        "        print(f'[!] Engine 12 weights not found for Fold {FOLD}')\\n"
        "        return None\\n"
        "\\n"
        "    print(f'[+] Loading Engine 12 Fold {FOLD} from: {weight_file}', flush=True)\\n"
        "\\n"
        "    bb = timm.create_model('convnext_large.fb_in22k_ft_in1k_384', pretrained=False, num_classes=0)\\n"
        "    bb.set_grad_checkpointing(True)\\n"
        "    model = ConvNeXtMILClassifier(bb, feat_dim=1536, num_targets=12).to(device)\\n"
        "\\n"
        "    ckpt = torch.load(weight_file, map_location=device, weights_only=False)\\n"
        "    state_dict = ckpt['model'] if 'model' in ckpt else ckpt\\n"
        "    model.load_state_dict(state_dict, strict=False)\\n"
        "    model.eval()\\n"
        "\\n"
        "    val_dataset = StudyWindowsDataset(\\n"
        "        ids=list(id2row.keys()),\\n"
        "        id2row=id2row,\\n"
        "        labels=val_labels,\\n"
        "        vols=all_vols,\\n"
        "        masks=None,  # Will be loaded from corpus if available\\n"
        "        res=384,\\n"
        "        k=16,\\n"
        "        train=False\\n"
        "    )\\n"
        "\\n"
        "    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)\\n"
        "\\n"
        "    predictions = []\\n"
        "    uids = []\\n"
        "\\n"
        "    print(f'[*] Running Engine 12 Fold {FOLD} inference on {len(val_dataset)} validation studies...', flush=True)\\n"
        "    with torch.no_grad():\\n"
        "        for windows, _, uid in val_loader:\\n"
        "            windows = windows.to(device)\\n"
        "            logits = model(windows)\\n"
        "            probs = torch.sigmoid(logits).cpu().numpy()[0]\\n"
        "            predictions.append(probs)\\n"
        "            uids.append(uid)\\n"
        "\\n"
        "    pred_df = pd.DataFrame(predictions, columns=TARGETS)\\n"
        "    pred_df.insert(0, 'StudyInstanceUID', uids)\\n"
        "    pred_df.to_csv(f'/content/engine12_fold{FOLD}_oof.csv', index=False)\\n"
        "    print(f'[+] Engine 12 Fold {FOLD} OOF saved: {pred_df.shape}', flush=True)\\n"
        "    return pred_df\\n"
        "\\n"
        "    # Train this fold\\n"
        "    print(f'=== Starting Engine 12 Fold {FOLD} Training ===', flush=True)\\n"
        "    exec(open(f'train_engine12_fold{FOLD}.py').read())\\n"
        "\\n"
        "    # Publish weights\\n"
        "    import os, json, subprocess, shutil\\n"
        "    os.makedirs(f'/content/engine12_fold{FOLD}_upload', exist_ok=True)\\n"
        "    shutil.copy(f'/content/engine12_convnext_large_fold{FOLD}_best.pt', f'/content/engine12_fold{FOLD}_upload/')\\n"
        "\\n"
        "    meta = {\\n"
        "        'title': f'RSNA Knee Engine 12 ConvNeXt-Large Fold {FOLD} Weights',\\n"
        "        'id': f'cyruspinto/rsna-knee-engine12-fold{FOLD}-weights',\\n"
        "        'licenses': [{'name': 'CC0-1.0'}]\\n"
        "    }\\n"
        "    with open(f'/content/engine12_fold{FOLD}_upload/dataset-metadata.json', 'w') as f:\\n"
        "        json.dump(meta, f, indent=2)\\n"
        "\\n"
        "    subprocess.run(['kaggle', 'datasets', 'create', '-p', f'/content/engine12_fold{FOLD}_upload', '-r', 'zip'], capture_output=True)\\n"
        "    print(f'[+] Engine 12 Fold {FOLD} weights pushed to Kaggle', flush=True)\\n"
        "\\n"
        "    # Clear GPU memory for next fold\\n"
        "    import torch, gc\\n"
        "    torch.cuda.empty_cache()\\n"
        "    gc.collect()\\n"
        "    print(f'=== Fold {FOLD} Complete ===', flush=True)\\n"
        "\\n"
        "print('=== All Engine 12 Folds Complete ===')\\n"""

# Build Engine 12 multi-fold notebook
with open(r'C:\Users\Cyrus\Downloads\colab_engine12_fold1_master.ipynb', 'r', encoding='utf-8') as f:
    engine12_base = json.load(f)

new_cells = engine12_base['cells'][:5]  # Keep cells 0-4
new_cells.append({
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': [engine12_multi]
})
new_cells.append({
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': ['print("=== All Engine 12 Folds Complete ===")']
})

new_nb = {
    'cells': engine12_base['cells'][:5] + [
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [engine12_multi]},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': ['print("=== All Engine 12 Folds Complete ===")']}
    ],
    'metadata': engine12_base['metadata'],
    'nbformat': engine12_base['nbformat'],
    'nbformat_minor': engine12_base['nbformat_minor']
}

with open(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb', 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, indent=1)
print('Created Engine 12 multi-fold')
"