import json
from pathlib import Path

def fix_notebook(engine_num):
    if engine_num == 12:
        src = Path(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb')
        dst = Path(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb')
        model_name = 'convnext_large.fb_in22k_ft_in1k_384'
        weight_prefix = 'engine12_convnext_large'
        model_class = 'ConvNeXtMILClassifier'
        feat_dim = 1536
        k_val = 16
        train_script_prefix = 'train_engine12_fold'
        upload_prefix = 'engine12_fold'
        weight_prefix = 'engine12_convnext_large'
        model_id = 'convnext_large.fb_in22k_ft_in1k_384'
        title_prefix = 'RSNA Knee Engine 12 ConvNeXt-Large'
        dataset_prefix = 'cyruspinto/rsna-knee-engine12-fold'
    else:
        src = Path(r'C:\Users\Cyrus\Downloads\colab_engine13_multi_fold_master.ipynb')
        dst = Path(r'C:\Users\Cyrus\Downloads\colab_engine13_multi_fold_master.ipynb')
        weight_prefix = 'engine13_swin_base'
        train_script_prefix = 'train_engine13_fold'
        upload_prefix = 'engine13_fold'
        weight_prefix = 'engine13_swin_base'
        model_id = 'swin_base_patch4_window12_384.ms_in22k_ft_in1k'
        title_prefix = 'RSNA Knee Engine 13 Swin-Base'
        dataset_prefix = 'cyruspinto/rsna-knee-engine13-fold'
        model_class = 'SwinMILClassifier'
        feat_dim = 1024
        k_val = 8

    if engine_num == 12:
        src = Path(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb')
        dst = Path(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb')
    else:
        src = Path(r'C:\Users\Cyrus\Downloads\colab_engine13_multi_fold_master.ipynb')
        dst = Path(r'C:\Users\Cyrus\Downloads\colab_engine13_multi_fold_master.ipynb')

    with open(src) as f:
        nb = json.load(f)

    # Find the multi-fold loop cell (cell 2)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'for FOLD in [2, 3, 4]:' in ''.join(cell['source']):
            # Fix the training script writer - replace the problematic f-string with triple quotes
            if engine_num == 12:
                new_source = engine12_fixed_source()
            else:
                new_source = engine13_fixed_source()
            
            nb['cells'][i]['source'] = [engine12_fixed_source() if engine_num == 12 else engine13_fixed_source()]
            break

    with open(dst, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'Fixed: {dst}')

def engine12_fixed_source():
    return '''# Multi-Fold Loop: Folds 2, 3, 4
for FOLD in [2, 3, 4]:
    print(f"=== Starting Fold {FOLD} ===", flush=True)
    
    # Write training script for this fold
    script_content = """ 
RSNA Knee Engine 12 - ConvNeXt-Large-384 Gated MIL (Fold {FOLD} of 5)
5-Fold Stratified Cross-Validation + Official 58 Gold Radiologist Anchoring
"""
import os, sys, time, math, random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import timm

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class TwoPartVols:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + (b.shape[0] if b is not None else 0),) + tuple(a.shape[1:])
    def __len__(self): return self.shape[0]
    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]

class StudyWindowsDataset(Dataset):
    def __init__(self, ids, id2row, labels, vols, masks, res=384, k=16, train=False):
        self.ids = ids
        self.id2row = id2row
        self.labels = labels
        self.vols = vols
        self.masks = masks
        self.res = res
        self.k = k
        self.train = train

    def __len__(self): return len(self.ids)

    def _centers(self, valid, count, d_len):
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs:
            cs = [max(1, min((lo + hi) // 2, d_len - 2))]
        if self.train:
            reps = count // len(cs) + 1
            pool = (cs * reps)
            random.shuffle(pool)
            return sorted(pool[:count])
        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)
        return sorted([cs[i] for i in idx])

    def __getitem__(self, i):
        uid = self.ids[i]
        row = self.id2row[uid]
        vol = self.vols[row]
        mask = self.masks[row] if self.masks is not None else None
        
        # Get valid slice indices
        if mask is not None:
            valid = np.where(mask)[0]
        else:
            valid = np.arange(vol.shape[0])
        
        centers = self._centers(valid, self.k, vol.shape[0])
        
        windows = []
        for c in centers:
            if c - 1 >= 0 and c + 1 < vol.shape[0]:
                window = vol[c-1:c+2]  # 3 slices
            else:
                window = np.zeros((3, vol.shape[1], vol.shape[2]), dtype=vol.dtype)
                mid = min(max(c, 1), vol.shape[0] - 2)
                window = vol[mid-1:mid+2]
            # Resize to 384x384
            from skimage.transform import resize
            window = np.stack([resize(w, (self.res, self.res), order=1, preserve_range=True, anti_aliasing=True) for w in window])
            windows.append(window)
        
        windows = np.stack(windows)  # (k, 3, H, W)
        windows = torch.from_numpy(windows).float() / 255.0
        
        # Labels
        y = self.labels.loc[self.labels['StudyInstanceUID'] == uid, TARGETS].values
        if len(y) == 0:
            y = np.zeros(len(TARGETS), dtype=np.float32)
        else:
            y = y[0].astype(np.float32)
        
        return windows, torch.from_numpy(y), uid


class ConvNeXtMILClassifier(nn.Module):
    def __init__(self, backbone, feat_dim=1536, num_targets=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.norm = nn.LayerNorm(feat_dim)
        self.att = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(256, num_targets)
        )
        self.clsW = nn.Parameter(torch.zeros(num_targets, feat_dim))
        self.clsb = nn.Parameter(torch.zeros(num_targets))

    def encode(self, x):
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        chunk_size = 4
        feats = []
        for i in range(0, flat_x.size(0), chunk_size):
            f = self.backbone(flat_x[i:i + chunk_size])
            feats.append(f)
        feats = torch.cat(feats, dim=0)
        return feats.view(B, K, -1)

    def forward(self, x):
        feats = self.encode(x)
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

def run_engine12_inference():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Find weight file
    weight_paths = [
        Path(f'/content/weights/engine12_convnext_large_fold{FOLD}_best.pt'),
        Path(f'/content/weights/engine12_convnext_large_fold{FOLD}_best.pt'),
        Path('/kaggle/input/rsna-knee-engine12-convnext-large-weights/engine12_convnext_large_best.pt'),
    ]
    weight_file = next((p for p in weight_paths if p.is_file()), None)
    
    if weight_file is None:
        print(f'[!] Engine 12 weights not found for Fold {FOLD}')
        return None
    
    print(f'[+] Loading Engine 12 Fold {FOLD} from: {weight_file}', flush=True)
    
    bb = timm.create_model('convnext_large.fb_in22k_ft_in1k_384', pretrained=False, num_classes=0)
    bb.set_grad_checkpointing(True)
    model = ConvNeXtMILClassifier(bb, feat_dim=1536, num_targets=12).to(device)
    
    ckpt = torch.load(weight_file, map_location=device, weights_only=False)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Run inference on validation set
    val_dataset = StudyWindowsDataset(
        ids=list(id2row.keys()),
        id2row=id2row,
        labels=val_labels,
        vols=all_vols,
        masks=None,  # Will be loaded from corpus if available
        res=384,
        k=16,
        train=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    predictions = []
    uids = []
    
    print(f'[*] Running Engine 12 Fold {FOLD} inference on {len(val_dataset)} validation studies...', flush=True)
    with torch.no_grad():
        for windows, _, uid in val_loader:
            windows = windows.to(device)
            logits = model(windows)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            predictions.append(probs)
            uids.append(uid)
    
    pred_df = pd.DataFrame(predictions, columns=TARGETS)
    pred_df.insert(0, 'StudyInstanceUID', uids)
    pred_df.to_csv(f'/content/engine12_fold{FOLD}_oof.csv', index=False)
    print(f'[+] Engine 12 Fold {FOLD} OOF saved: {pred_df.shape}', flush=True)
    return pred_df

    # Train this fold
    print(f'=== Starting Engine 12 Fold {FOLD} Training ===', flush=True)
    exec(open(f'train_engine12_fold{FOLD}.py').read())
    
    # Publish weights
    import os, json, subprocess, shutil
    os.makedirs(f'/content/engine12_fold{FOLD}_upload', exist_ok=True)
    shutil.copy(f'/content/engine12_convnext_large_fold{FOLD}_best.pt', f'/content/engine12_fold{FOLD}_upload/')
    
    meta = {
        'title': f'RSNA Knee Engine 12 ConvNeXt-Large Fold {FOLD} Weights',
        'id': f'cyruspinto/rsna-knee-engine12-fold{FOLD}-weights',
        'licenses': [{'name': 'CC0-1.0'}]
    }
    with open(f'/content/engine12_fold{FOLD}_upload/dataset-metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    subprocess.run(['kaggle', 'datasets', 'create', '-p', f'/content/engine12_fold{FOLD}_upload', '-r', 'zip'], capture_output=True)
    print(f'[+] Engine 12 Fold {FOLD} weights pushed to Kaggle', flush=True)
    
    # Clear GPU memory for next fold
    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()
    print(f'=== Fold {FOLD} Complete ===', flush=True)

print('=== All Engine 12 Folds Complete ===')
"""

def engine13_fixed_source():
    return '''# Multi-Fold Loop: Folds 2, 3, 4
for FOLD in [2, 3, 4]:
    print(f"=== Starting Fold {FOLD} ===", flush=True)
    
    # Write training script for this fold
    script_content = """
RSNA Knee Engine 13 - Swin-Base-384 Focal MIL (Fold {FOLD} of 5)
5-Fold Stratified Cross-Validation + Official 58 Gold Radiologist Anchoring
"""
import os, sys, time, math, random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import timm

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class TwoPartVols:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + (b.shape[0] if b is not None else 0),) + tuple(a.shape[1:])
    def __len__(self): return self.shape[0]
    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]

class SwinMILClassifier(nn.Module):
    def __init__(self, backbone, feat_dim=1024, num_targets=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.norm = nn.LayerNorm(feat_dim)
        self.att = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(256, num_targets)
        )
        self.clsW = nn.Parameter(torch.zeros(num_targets, feat_dim))
        self.clsb = nn.Parameter(torch.zeros(num_targets))

    def encode(self, x):
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        chunk_size = 4
        feats = []
        for i in range(0, flat_x.size(0), chunk_size):
            f = self.backbone(flat_x[i:i + chunk_size])
            feats.append(f)
        feats = torch.cat(feats, dim=0)
        return feats.view(B, K, -1)

    def forward(self, x):
        feats = self.encode(x)
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

def run_engine13_inference():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    weight_paths = [
        Path(f'/content/weights/engine13_swin_base_fold{FOLD}_best.pt'),
        Path(f'/content/weights/engine13_swin_base_fold{FOLD}_best.pt'),
        Path('/kaggle/input/rsna-knee-engine13-swin-base-weights/engine13_swin_base_best.pt'),
    ]
    weight_file = next((p for p in weight_paths if p.is_file()), None)
    
    if weight_file is None:
        print(f'[!] Engine 13 weights not found for Fold {FOLD}')
        return None
    
    print(f'[+] Loading Engine 13 Fold {FOLD} from: {weight_file}', flush=True)
    
    bb = timm.create_model('swin_base_patch4_window12_384.ms_in22k_ft_in1k', pretrained=False, num_classes=0)
    model = SwinMILClassifier(bb, feat_dim=1024, num_targets=12).to(device)
    
    ckpt = torch.load(weight_file, map_location=device, weights_only=False)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    val_dataset = StudyWindowsDataset(
        ids=list(id2row.keys()),
        id2row=id2row,
        labels=val_labels,
        vols=all_vols,
        masks=None,
        res=384,
        k=8,
        train=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    predictions = []
    uids = []
    
    print(f'[*] Running Engine 13 Fold {FOLD} inference on {len(val_dataset)} validation studies...', flush=True)
    with torch.no_grad():
        for windows, _, uid in val_loader:
            windows = windows.to(device)
            logits = model(windows)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            predictions.append(probs)
            uids.append(uid)
    
    pred_df = pd.DataFrame(predictions, columns=TARGETS)
    pred_df.insert(0, 'StudyInstanceUID', uids)
    pred_df.to_csv(f'/content/engine13_fold{FOLD}_oof.csv', index=False)
    print(f'[+] Engine 13 Fold {FOLD} OOF saved: {pred_df.shape}', flush=True)
    return pred_df

    # Train this fold
    print(f'=== Starting Engine 13 Fold {FOLD} Training ===', flush=True)
    exec(open(f'train_engine13_fold{FOLD}.py').read())
    
    # Publish weights
    import os, json, subprocess, shutil
    os.makedirs(f'/content/engine13_fold{FOLD}_upload', exist_ok=True)
    shutil.copy(f'/content/engine13_swin_base_fold{FOLD}_best.pt', f'/content/engine13_fold{FOLD}_upload/')
    
    meta = {
        'title': f'RSNA Knee Engine 13 Swin-Base Fold {FOLD} Weights',
        'id': f'cyruspinto/rsna-knee-engine13-fold{FOLD}-weights',
        'licenses': [{'name': 'CC0-1.0'}]
    }
    with open(f'/content/engine13_fold{FOLD}_upload/dataset-metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    subprocess.run(['kaggle', 'datasets', 'create', '-p', f'/content/engine13_fold{FOLD}_upload', '-r', 'zip'], capture_output=True)
    print(f'[+] Engine 13 Fold {FOLD} weights pushed to Kaggle', flush=True)
    
    # Clear GPU memory for next fold
    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()
    print(f'=== Fold {FOLD} Complete ===', flush=True)

print('=== All Engine 13 Folds Complete ===')
'''

if __name__ == '__main__':
    import json
    from pathlib import Path
    
    # Fix Engine 12
    src = Path(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb')
    with open(src) as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'for FOLD in [2, 3, 4]:' in ''.join(cell['source']):
            nb['cells'][i]['source'] = [engine12_fixed_source()]
            break
    
    with open(Path(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb'), 'w') as f:
        json.dump(nb, f, indent=1)
    print('Fixed Engine 12 notebook')
    
    # Fix Engine 13
    src = Path(r'C:\Users\Cyrus\Downloads\colab_engine13_multi_fold_master.ipynb')
    with open(src) as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'for FOLD in [2, 3, 4]:' in ''.join(cell['source']):
            nb['cells'][i]['source'] = [engine13_fixed_source()]
            break
    
    with open(Path(r'C:\Users\Cyrus\Downloads\colab_engine13_multi_fold_master.ipynb'), 'w') as f:
        json.dump(nb, f, indent=1)
    print('Fixed Engine 13 notebook')
"