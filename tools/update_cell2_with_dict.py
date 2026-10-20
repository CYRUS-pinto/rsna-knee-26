import json

with open('tools/gold_dict.py') as f:
    dict_code = f.read().strip()

nb_path = 'colab_swin_single_click_master.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell2_source = f"""# CELL 2: Autonomous Swin-Base-384 Training Loop (Fold 2 & Fold 3)
import os, sys, math, time, gc, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class TwoPartVols:
    def __init__(self, a, b=None):
        self.a, self.b = a, b
        self.len_a = len(a)
    def __getitem__(self, idx):
        if idx < self.len_a: return self.a[idx]
        return self.b[idx - self.len_a]
    def __len__(self):
        return self.len_a + (len(self.b) if self.b is not None else 0)

class SwinMILDataset(Dataset):
    def __init__(self, uids, id2row, labels, vols, res=384, train=True, n_slices=6):
        self.uids = uids
        self.id2row = id2row
        self.labels = labels
        self.vols = vols
        self.res = res
        self.train = train
        self.n_slices = n_slices
    def __len__(self): return len(self.uids)
    def __getitem__(self, idx):
        u = self.uids[idx]
        row = self.id2row[u]
        v = self.vols[row]
        D, H, W = v.shape
        step = max(1, D // self.n_slices)
        idxs = [min(i * step + step // 2, D - 1) for i in range(self.n_slices)]
        imgs = []
        for ci in idxs:
            c_prev = max(0, ci - 1)
            c_next = min(D - 1, ci + 1)
            triplet = np.stack([v[c_prev], v[ci], v[c_next]], axis=0).astype(np.float32) / 255.0
            imgs.append(triplet)
        t = torch.tensor(np.stack(imgs, axis=0), dtype=torch.float32)
        if (H, W) != (self.res, self.res):
            t = nn.functional.interpolate(t, size=(self.res, self.res), mode='bilinear', align_corners=False)
        y = torch.tensor(self.labels[u], dtype=torch.float32)
        return t, y

class SwinMILModel(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window12_384.ms_in22k_ft_in1k', num_classes=12):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=True, num_classes=0)
        d = self.encoder.num_features
        self.attn = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Linear(d, num_classes)
    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.encoder(x)
        feats = feats.view(B, S, -1)
        scores = self.attn(feats)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(feats * weights, dim=1)
        return self.classifier(pooled)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Training on Device: {{DEVICE}}', flush=True)

# 1. Load 384px corpus
va = np.load('/content/corpus/all_vols.npy', mmap_mode='r')
vb = np.load('/content/corpus/extra_vols.npy', mmap_mode='r') if os.path.exists('/content/corpus/extra_vols.npy') else None
vols = TwoPartVols(va, vb)
a_ids = np.load('/content/corpus/all_ids.npy', allow_pickle=True).astype(str)
b_ids = np.load('/content/corpus/extra_ids.npy', allow_pickle=True).astype(str) if os.path.exists('/content/corpus/extra_ids.npy') else np.array([])
all_ids = np.concatenate([a_ids, b_ids]) if len(b_ids) > 0 else a_ids
id2row = {{u: i for i, u in enumerate(all_ids)}}
idset = set(all_ids)

# 2. Native Python Ground Truth Dictionary (Zero CSV/parser errors)
{dict_code}
gold_ids = [u for u in GOLD_LABELS.keys() if u in idset]
print(f'[+] Gold-58 labels matched: {{len(gold_ids)}}/{{len(GOLD_LABELS)}}', flush=True)

# 3. Load Super Consensus Soft Labels
soft = pd.read_parquet('/content/labels/SUPER_CONSENSUS_labels.parquet').set_index('StudyInstanceUID')
train_ids = [u for u in all_ids if u in soft.index and u not in set(gold_ids)]
labels = {{u: soft.loc[u, TARGETS].values.astype(np.float32) for u in train_ids}}
for u in gold_ids:
    labels[u] = np.array(GOLD_LABELS[u], dtype=np.float32)

print(f'[+] Total training pool: {{len(train_ids)}} soft + {{len(gold_ids)}} gold = {{len(labels)}} total', flush=True)

np.random.seed(42)
shuffled_train = list(train_ids)
np.random.shuffle(shuffled_train)
n_val = len(shuffled_train) // 5

FOLDS_TO_TRAIN = [2, 3]
EPOCHS = 6
BATCH_SIZE = 8

for fold in FOLDS_TO_TRAIN:
    print(f'\\n' + '='*60)
    print(f'=== STARTING ENGINE 10 SWIN-BASE-384: FOLD {{fold}} ===')
    print('='*60, flush=True)
    val_ids = shuffled_train[fold * n_val : (fold + 1) * n_val]
    core_train_ids = [u for u in shuffled_train if u not in set(val_ids)]
    
    tds = SwinMILDataset(core_train_ids, id2row, labels, vols, res=384, train=True, n_slices=6)
    tl = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    model = SwinMILModel().to(DEVICE)
    optimizer = torch.optim.AdamW([
        {{'params': model.encoder.parameters(), 'lr': 3e-5, 'weight_decay': 1e-4}},
        {{'params': model.attn.parameters(), 'lr': 3e-4, 'weight_decay': 1e-4}},
        {{'params': model.classifier.parameters(), 'lr': 3e-4, 'weight_decay': 1e-4}}
    ])
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        start_t = time.time()
        for step, (bx, by) in enumerate(tl):
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(bx)
                loss = criterion(logits, by)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if (step + 1) % 50 == 0:
                print(f'Fold {{fold}} | Epoch {{epoch+1}}/{{EPOCHS}} | Step {{step+1}}/{{len(tl)}} | Loss: {{total_loss/(step+1):.4f}}', flush=True)
        elapsed_epoch = time.time() - start_t
        print(f'Epoch {{epoch+1}} finished in {{elapsed_epoch:.1f}}s. Avg Loss: {{total_loss/len(tl):.4f}}', flush=True)
        
    weight_path = f'/content/engine10_swin_384_fold{{fold}}_best.pt'
    torch.save(model.state_dict(), weight_path)
    print(f'[+] Saved model weights to {{weight_path}}', flush=True)
    
    # Export directly to Kaggle Datasets via API
    print(f'Pushing Fold {{fold}} weights to Kaggle Datasets...', flush=True)
    pkg_dir = f'/content/kaggle_engine10_fold{{fold}}'
    os.makedirs(pkg_dir, exist_ok=True)
    os.system(f'cp {{weight_path}} {{pkg_dir}}/')
    meta = {{
        'title': f'RSNA Knee Engine 10 Swin-Base-384 Fold {{fold}} Weights',
        'id': f'cyruspinto/rsna-knee-engine10-fold{{fold}}-weights',
        'licenses': [{{'name': 'CC0-1.0'}}]
    }}
    with open(f'{{pkg_dir}}/dataset-metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    try:
        api.dataset_create_new(pkg_dir, dir_mode='zip')
        print(f'[+] Created new dataset: cyruspinto/rsna-knee-engine10-fold{{fold}}-weights')
    except Exception as e:
        print(f'Dataset exists, creating version update...')
        api.dataset_create_version(pkg_dir, f'Update Fold {{fold}} Weights', dir_mode='zip')
        print(f'[+] Updated dataset: cyruspinto/rsna-knee-engine10-fold{{fold}}-weights')
    
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

print('\\n' + '='*60)
print('=== TRAINING AND UPLOAD COMPLETE! ===')
print('='*60, flush=True)
"""

nb['cells'][2]['source'] = [line + '\n' for line in cell2_source.split('\n')]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Updated colab_swin_single_click_master.ipynb with Python dictionary!')
