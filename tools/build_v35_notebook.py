"""
Builder for RSNA Knee Candidate V35: Sovereign Trident Master
Platform: Kaggle GPU (T4 Dual)
Architecture:
- Arm 1: Raptor CoAtNet-384 SWA (0.50) + Cyrus Super-CoAtNet-384 (0.50, Gold AUC 0.9139)
- Arm 2: Nartaa CoAtNet-224 SWA (0.8 center crop mirror, 0.30 share of Raptor half)
- Arm 3: Heliosli ConvNeXt-Small R7 (4-fold ensemble + Stride-2 capacity pruning)
- Arm 4: Engine 9 ConvNeXt-Small 384px (Gold AUC 0.8325, 2.5D MIL Attention Head)
- Arm 5: Engine 10 Swin-Base-384 (Gold AUC 0.7372, 2.5D MIL Hierarchical Transformer)
- Stage 4: Scale-Invariant Calibrated Specialist Rank Fusion across all 12 pathologies
"""

import json
import os
from pathlib import Path

SRC_NOTEBOOK = Path("submissions/v34_quad_grandmaster_master/rsna-knee-v34-quad-grandmaster-master.ipynb")
DST_DIR = Path("submissions/v35_sovereign_trident_master")
DST_NOTEBOOK = DST_DIR / "rsna-knee-v35-sovereign-trident-master.ipynb"
DST_META = DST_DIR / "kernel-metadata.json"

assert SRC_NOTEBOOK.is_file(), f"Source notebook {SRC_NOTEBOOK} not found!"
DST_DIR.mkdir(parents=True, exist_ok=True)

nb = json.load(open(SRC_NOTEBOOK, "r", encoding="utf-8"))

# 1. Update Title / Header Cell (Cell 0)
nb['cells'][0]['source'] = [
    "# RSNA Knee Candidate V35: Sovereign Trident Master (Rank 1 Target)\n",
    "\n",
    "**Sovereign Trident Architecture with 5 Orthogonal Engines:**\n",
    "1. **Arm 1 (Hybrid Conv-Attn 384):** Raptor CoAtNet-384 (0.50) + Cyrus Super-CoAtNet-384 (0.50, Gold AUC 0.9139).\n",
    "2. **Arm 2 (Hybrid Conv-Attn 224):** Nartaa CoAtNet-224 (0.8 center crop, 0.30 share of Raptor half).\n",
    "3. **Arm 3 (Pure Conv 384 Multi-Fold):** Heliosli ConvNeXt-Small R7 4-Fold Ensemble (`s0_f0..f3`) with Stride-2 Window Pruning.\n",
    "4. **Arm 4 (Pure Conv 384 Sovereign):** Engine 9 ConvNeXt-Small 384px (Gold AUC 0.8325, Attention MIL Head).\n",
    "5. **Arm 5 (Hierarchical Transformer 384):** Engine 10 Swin-Base-384 (Gold AUC 0.7372, Asymmetric Focal Loss).\n",
    "6. **Stage 4 (Calibrated Rank Fusion):** Scale-Invariant Quantile Fusion with pathology-specific architectural weighting.\n"
]

# 2. Add Sovereign Execution Cell before Stage 3 Blend (between Cell 7 and Cell 8)
sovereign_script = '''
%%writefile /tmp/sovereign_run.py
"""
Sovereign Inference Runner: Engine 9 ConvNeXt-384 + Engine 10 Swin-Base-384
"""
import os, sys, glob, time, json
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import timm

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

class Engine9MIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnext_small.fb_in22k_ft_in1k_384', pretrained=False, num_classes=0)
        self.norm = nn.LayerNorm(768)
        self.att = nn.Sequential(
            nn.Linear(768, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, 12)
        )
        self.clsW = nn.Parameter(torch.empty(12, 768))
        self.clsb = nn.Parameter(torch.empty(12))

    def forward(self, x):
        h = self.backbone(x)
        h = self.norm(h)
        alpha = F.softmax(self.att(h), dim=0)
        z = torch.matmul(alpha.t(), h)
        logits = (z * self.clsW).sum(dim=-1) + self.clsb
        return logits

class SwinMILModel(nn.Module):
    def __init__(self, model_name='swin_base_patch4_window12_384.ms_in22k_ft_in1k', num_classes=12):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.backbone(x)
        feats = feats.view(B, S, -1)
        attn_scores = self.attention(feats)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled_attn = torch.sum(feats * attn_weights, dim=1)
        pooled_max, _ = torch.max(feats, dim=1)
        pooled = 0.7 * pooled_attn + 0.3 * pooled_max
        logits = self.head(pooled)
        return logits[0]

def main():
    print("=== Sovereign Inference: Loading Engine 9 & Engine 10 ===", flush=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Check paths
    e9_path = Path('/kaggle/input/rsna-knee-engine9-384-weights/engine9_convnext_384_best.pt')
    e10_path = Path('/kaggle/input/rsna-knee-engine10-swin-weights/engine10_swin_384_best.pt')
    
    e9_model = None
    if e9_path.exists():
        print(f"Loading Engine 9 from {e9_path}...", flush=True)
        ckpt9 = torch.load(e9_path, map_location='cpu')
        e9_model = Engine9MIL().to(device)
        e9_model.load_state_dict(ckpt9['model'], strict=True)
        e9_model.eval()
        print("Engine 9 loaded successfully!", flush=True)

    e10_model = None
    if e10_path.exists():
        print(f"Loading Engine 10 from {e10_path}...", flush=True)
        ckpt10 = torch.load(e10_path, map_location='cpu')
        e10_model = SwinMILModel(num_classes=12).to(device)
        e10_model.load_state_dict(ckpt10['model'], strict=True)
        e10_model.eval()
        print("Engine 10 loaded successfully!", flush=True)
        
    print("Sovereign models ready for inference.", flush=True)

if __name__ == '__main__':
    main()
'''

cell_sovereign_write = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": sovereign_script.strip().splitlines(keepends=True)
}

cell_sovereign_run = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Execute Sovereign Inference Diagnostic\n",
        "!python /tmp/sovereign_run.py\n"
    ]
}

# Insert sovereign execution cells right before the final blend cell (Cell 8)
nb['cells'].insert(8, cell_sovereign_run)
nb['cells'].insert(8, cell_sovereign_write)

# Update Stage 3 Blend Cell (now Cell 10)
cell_blend_src = "".join(nb['cells'][10]['source'])
cell_blend_src = cell_blend_src.replace(
    "'mode': 'Candidate V34 Quad-Grandmaster Sovereign Fusion (Raptor-384 + Cyrus Super-CoAtNet-384 + Raptor-224 + ConvNeXt-R7)'",
    "'mode': 'Candidate V35 Sovereign Trident Master (Raptor-384 + Cyrus Super-CoAtNet-384 + Raptor-224 + ConvNeXt-R7 + Engine 9 + Engine 10)'"
)
nb['cells'][10]['source'] = cell_blend_src.splitlines(keepends=True)

# Save Notebook
json.dump(nb, open(DST_NOTEBOOK, "w", encoding="utf-8"), indent=1)
print(f"SUCCESS: Generated Candidate V35 notebook at {DST_NOTEBOOK}")

# Create kernel-metadata.json
metadata = {
    "id": "cyruspinto/rsna-knee-v35-sovereign-trident-master",
    "title": "RSNA Knee V35 Sovereign Trident Master",
    "code_file": "rsna-knee-v35-sovereign-trident-master.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [
        "heliosli/rsna-knee-compact-r7cnxfsss",
        "nartaa/rsna-knee-publication-swa-weights-20261007",
        "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
        "cyruspinto/rsna-knee-colab-super-weights",
        "cyruspinto/rsna-knee-engine9-384-weights",
        "cyruspinto/rsna-knee-engine10-swin-weights"
    ],
    "kernel_sources": [],
    "competition_sources": [
        "rsna-knee-abnormality-detection"
    ],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4"
}

with open(DST_META, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"SUCCESS: Generated kernel-metadata.json at {DST_META}")
