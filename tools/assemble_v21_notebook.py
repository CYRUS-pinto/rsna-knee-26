import json
import os

def create_engine5_stage_code():
    code = '''%%writefile /tmp/engine5_run.py
import os
import sys
import time
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

ALL_LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

SPECIALIST_TARGETS = {
    'sagittal': ['ACL', 'Medial Meniscus', 'Lateral Meniscus', 'Contusion'],
    'coronal': ['MCL', 'Medial OA', 'Lateral OA'],
    'axial': ['PF OA', 'Effusion', 'Synovitis', "Baker's", 'Fracture']
}

PLANE_SLOTS = {
    'sagittal': [(0, 26, 8), (26, 48, 8)],     # 16 slices total
    'coronal': [(48, 66, 8), (66, 78, 8)],     # 16 slices total
    'axial': [(78, 96, 10), (0, 26, 6)]        # 16 slices total
}

class GatedAttentionPoolingHead(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim=256, drop=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(in_features)
        self.attention_V = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh(), nn.Dropout(drop))
            for _ in range(num_classes)
        ])
        self.attention_U = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid(), nn.Dropout(drop))
            for _ in range(num_classes)
        ])
        self.attention_weights = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_classes)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(in_features, 1) for _ in range(num_classes)
        ])
        
    def forward(self, feats):
        B, K, D = feats.shape
        h = self.norm(feats)
        logits_list = []
        for j in range(self.num_classes):
            v = self.attention_V[j](h)
            u = self.attention_U[j](h)
            scores = self.attention_weights[j](v * u).squeeze(-1)
            attn = torch.softmax(scores, dim=-1)
            pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
            logit = self.classifiers[j](pooled)
            logits_list.append(logit)
        return torch.cat(logits_list, dim=-1)

class SpecialistMIL(nn.Module):
    def __init__(self, backbone_name='convnext_small', num_classes=4):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, 
            in_chans=3, global_pool='avg'
        )
        feat_dim = self.backbone.num_features
        self.head = GatedAttentionPoolingHead(in_features=feat_dim, num_classes=num_classes)
        
    def forward(self, x):
        B, K, C, H, W = x.shape
        flat_x = x.view(B * K, C, H, W)
        feats = self.backbone(flat_x).view(B, K, -1)
        return self.head(feats)

def extract_plane_triples(vol_96, plane, res=224):
    slots = PLANE_SLOTS[plane]
    slabs = []
    import cv2
    for start, end, count in slots:
        max_c = end - 1
        steps = np.linspace(start, end - 1, count).round().astype(int)
        for c in steps:
            c = max(1, min(c, max_c - 1))
            tri = np.stack([vol_96[c - 1], vol_96[c], vol_96[c + 1]], axis=0).astype(np.float32) / 255.0
            if tri.shape[1] != res or tri.shape[2] != res:
                chans = [cv2.resize(tri[ch], (res, res), interpolation=cv2.INTER_LINEAR) for ch in range(3)]
                tri = np.stack(chans, axis=0)
            slabs.append(tri)
    return np.stack(slabs[:16], axis=0)

def main():
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Stage 2b] Engine 5 Specialists starting on {device}...")
    
    # Locate weights from mounted kernel
    weight_dir = None
    for cand in [
        '/kaggle/input/rsna-knee-engine5-decoupled-train/weights',
        '/kaggle/input/datasets/cyruspinto/rsna-knee-engine5-decoupled-train/weights',
        'training/engine5_decoupled/kaggle_output/weights'
    ]:
        if os.path.isdir(cand):
            weight_dir = Path(cand)
            break
            
    if weight_dir is None:
        raise FileNotFoundError("Could not find Engine 5 weights directory")
        
    # Load models
    models = {}
    for plane in ['sagittal', 'coronal', 'axial']:
        num_classes = len(SPECIALIST_TARGETS[plane])
        m = SpecialistMIL(num_classes=num_classes).to(device)
        w_path = weight_dir / f"engine5_{plane}_fold0_best.pth"
        sd = torch.load(w_path, map_location=device, weights_only=True)
        m.load_state_dict(sd)
        m.eval()
        models[plane] = m
        print(f"Loaded {plane} specialist weights from {w_path}")

    # Read test study list
    test_csv = Path('/kaggle/input/rsna-knee-abnormality-detection/test.csv')
    if not test_csv.is_file():
        test_csv = Path('test.csv')
    df_test = pd.read_csv(test_csv)
    test_uids = df_test['StudyInstanceUID'].astype(str).tolist()
    print(f"Evaluating {len(test_uids)} test studies...")

    # Load fastread builder
    import importlib.util as _iu
    fr_path = '/kaggle/working/_fastread.py' if os.path.exists('/kaggle/working/_fastread.py') else '_fastread.py'
    spec = _iu.spec_from_file_location('_fastread', fr_path)
    fr = _iu.module_from_spec(spec)
    spec.loader.exec_module(fr)
    
    # Fastread setup matching Raptor
    SLOTS = [
        ('Sagittal', 1, 26),
        ('Sagittal', 0, 22),
        ('Coronal',  1, 18),
        ('Coronal',  0, 12),
        ('Axial',    0, 18),
    ]
    reader = fr.make_builder('study_lut', 384, SLOTS, hdr_threads=8, px_threads=16)

    # Load series records
    tsdir = '/kaggle/input/rsna-knee-abnormality-detection/test_series'
    tseries_csv = Path('/kaggle/input/rsna-knee-abnormality-detection/test_series.csv')
    df_series = pd.read_csv(tseries_csv)
    ser_records = {}
    for _, r in df_series.iterrows():
        ser_records.setdefault(str(r['StudyInstanceUID']), []).append(r.to_dict())

    results = []
    with torch.no_grad():
        for sid in test_uids:
            # Build 96-slice volume
            rows = ser_records.get(sid, [])
            vol = np.zeros((96, 384, 384), np.uint8)
            idx = 0
            used = set()
            order_and_meta, read_px, _ = reader
            
            for plane, fluid, k in SLOTS:
                matched_r = None
                for r in rows:
                    if r['SeriesInstanceUID'] in used: continue
                    r_plane = str(r.get('SeriesDescription', '')).lower()
                    if plane.lower() in r_plane:
                        matched_r = r; break
                if matched_r is None and rows:
                    for r in rows:
                        if r['SeriesInstanceUID'] not in used:
                            matched_r = r; break
                if matched_r is None:
                    idx += k; continue
                used.add(matched_r['SeriesInstanceUID'])
                sdir = f"{tsdir}/{sid}/{matched_r['SeriesInstanceUID']}"
                files, med_ps = order_and_meta(sdir)
                if not files:
                    idx += k; continue
                n = len(files)
                lo, hi = int(n * 0.02), max(int(n * 0.02), int(n * 0.98) - 1)
                picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
                sel = [files[min(p, n - 1)] for p in picks]
                for fp, ps in sel:
                    if idx >= 96: break
                    try:
                        px = read_px(fp)
                        if px is not None:
                            loq, hiq = np.percentile(px, [2.0, 98.0])
                            aw = np.clip((px.astype(np.float32) - loq) / max(hiq - loq, 1e-6), 0, 1)
                            vol[idx] = (aw * 255.0).astype(np.uint8)
                    except Exception:
                        pass
                    idx += 1

            # Predict across specialists
            study_preds = {'StudyInstanceUID': sid}
            for plane in ['sagittal', 'coronal', 'axial']:
                triples = extract_plane_triples(vol, plane, res=224) # (16, 3, 224, 224)
                x = torch.from_numpy(triples).unsqueeze(0).to(device) # (1, 16, 3, 224, 224)
                with torch.amp.autocast('cuda') if device == 'cuda' else torch.no_grad():
                    logits = models[plane](x)
                    probs = torch.sigmoid(logits)[0].cpu().numpy()
                for j, target in enumerate(SPECIALIST_TARGETS[plane]):
                    study_preds[target] = float(probs[j])
            results.append(study_preds)

    df_out = pd.DataFrame(results)
    out_path = Path('/kaggle/working/engine5_sub.csv')
    df_out.to_csv(out_path, index=False)
    print(f"[Stage 2b] Complete! Wrote {len(df_out)} predictions to {out_path} in {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
'''
    return code

def main():
    v20_path = 'submissions/v20_power_rank_master/rsna-knee-v20-power-rank-blend.ipynb'
    with open(v20_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Title in Cell 0
    cell0_text = (
        "# RSNA Knee: Candidate V21 Quad-Engine Multi-Plane Specialist Blend\n"
        "# Built on the verified V17 (0.958 PB) + V20 Power-Rank architecture.\n"
        "# Upgrades:\n"
        "# 1. Adds Stage 2b: Engine 5 Plane-Decoupled Specialist Models (ADR-015, ADR-020).\n"
        "# 2. Uses 2.5D slice triples (c-1, c, c+1) to preserve volumetric continuity.\n"
        "# 3. Axial fluid specialist provides high-confidence signal for Synovitis (0.8917 AUC) and Effusion (0.8138 AUC).\n"
        "# 4. Retains Generalized Power-Mean (p=1.4) on acute tear findings.\n"
    )
    nb['cells'][0]['source'] = [line + '\n' for line in cell0_text.strip().split('\n')]

    # 2. Add Stage 2b Code Cell and Execution Cell
    stage2b_code = create_engine5_stage_code()
    stage2b_exec = (
        "# Stage 2b Execution: Engine 5 Plane-Decoupled Specialist Models\n"
        "import subprocess, sys\n"
        "res = subprocess.run([sys.executable, '/tmp/engine5_run.py'], capture_output=True, text=True)\n"
        "print(res.stdout)\n"
        "if res.returncode != 0:\n"
        "    print('STDERR:', res.stderr)\n"
        "    raise RuntimeError(f'Engine 5 execution failed with exit code {res.returncode}')\n"
    )

    cell_2b_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in stage2b_code.strip().split('\n')]
    }

    cell_2b_exec = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in stage2b_exec.strip().split('\n')]
    }

    # 3. Update Stage 3 Ensemble in Cell 8 to blend 4 engines
    # Insert Stage 2b before Stage 3
    nb['cells'].insert(8, cell_2b_code)
    nb['cells'].insert(9, cell_2b_exec)

    # Now Cell 10 is the Stage 3 ensemble cell
    # Update Cell 10 to include engine5_sub.csv
    stage3_src = ''.join(nb['cells'][10]['source'])
    stage3_src = stage3_src.replace(
        "ours = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})",
        "ours = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})\n"
        "e5 = pd.read_csv(WORK / 'engine5_sub.csv', dtype={'StudyInstanceUID': str})\n"
        "e5 = e5.set_index('StudyInstanceUID').loc[test_ids, labels]"
    )
    
    # Update blending weights:
    # Blend: b = 0.40 * raptor + 0.40 * ours + 0.20 * e5
    old_blend = "b = wr * rr + (1.0 - wr) * ro"
    new_blend = (
        "# Quad-Engine Blend: Raptor (40%), ConvNeXt R6 (40%), Engine 5 Specialist (20%)\n"
        "        re5 = pct(e5[c].values)\n"
        "        b = 0.40 * rr + 0.40 * ro + 0.20 * re5"
    )
    stage3_src = stage3_src.replace(old_blend, new_blend)

    old_power = "b = (wr * (rr_clean ** p) + (1.0 - wr) * (ro_clean ** p)) ** (1.0 / p)"
    new_power = (
        "re5 = pct(e5[c].values)\n"
        "        re5_clean = np.clip(re5, 1e-6, 1.0)\n"
        "        b = (0.40 * (rr_clean ** p) + 0.40 * (ro_clean ** p) + 0.20 * (re5_clean ** p)) ** (1.0 / p)"
    )
    stage3_src = stage3_src.replace(old_power, new_power)
    stage3_src = stage3_src.replace("Candidate V20 Generalized Power-Rank", "Candidate V21 Quad-Engine Multi-Plane Specialist Blend")
    stage3_src = stage3_src.replace("mode': 'blend_v20_power_rank'", "mode': 'blend_v21_quad_engine_specialist'")

    nb['cells'][10]['source'] = [line + '\n' for line in stage3_src.strip().split('\n')]

    out_path = 'submissions/v21_quad_engine_specialist/rsna-knee-v21-quad-engine-blend.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print(f"Assembled Candidate V21 notebook successfully: {out_path} ({len(nb['cells'])} cells)")

if __name__ == '__main__':
    main()
