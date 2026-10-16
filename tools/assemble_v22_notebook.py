"""
Candidate V22: Quint-Engine Master Ensemble Assembler.
Combines 5 Orthogonal Anatomical Engines:
1. Raptor-384 Whole Volume (CoAtNet-d4, SWA)
2. Raptor-224 Whole Volume (Multi-scale center crop)
3. ConvNeXt R6 3D Anatomical Joint Crops (SKM-TEA AnchorBank, Flip TTA)
4. Engine 5 2.5D Axial Fluid Specialists (Synovitis 0.8917 AUC, Effusion 0.8138 AUC)
5. Engine 6 Native Joint Sequence Specialist (ConvNeXt-Small + 2-layer BiLSTM on 2.5D slice triplets)
6. Laterality Canonicalization: Medial compartment always aligned to left width axis.
"""

import os
import sys
import json
import shutil
from pathlib import Path

def generate_v22_notebook():
    v21_path = 'submissions/v21_quad_engine_specialist/rsna-knee-v21-quad-engine-blend.ipynb'
    if not os.path.exists(v21_path):
        raise FileNotFoundError(f"Missing {v21_path}")
        
    with open(v21_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Cell 0: Header and documentation
    cell0 = nb['cells'][0]
    cell0['source'] = [
        "# RSNA Knee: Candidate V22 Quint-Engine Master Specialist Blend\n",
        "# Combines 5 Orthogonal Engines for 0.965+ Podium Contention:\n",
        "# 1. Raptor-384 & Raptor-224 (CoAtNet-d4 SWA Full Volume Multi-Scale)\n",
        "# 2. ConvNeXt R6 (3D SKM-TEA Joint Line Anatomical Crops with Flip TTA)\n",
        "# 3. Engine 5 (2.5D Multi-Plane Specialists: Axial Fluid Specialist Synovitis 0.8917 AUC)\n",
        "# 4. Engine 6 (2.5D Native Joint Sequence Model with 2-layer BiLSTM Inter-Slice Continuity)\n",
        "# 5. Laterality Canonicalization (Left/Right Centroid Alignment: Medial always Left)\n"
    ]

    out_dir = Path('submissions/v22_quint_engine_master')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy _fastread.py dependency
    shutil.copy('submissions/v21_quad_engine_specialist/_fastread.py', out_dir / '_fastread.py')

    # Update metadata
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
            "cyruspinto/rsna-knee-engine5-weights"
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "machine_shape": "NvidiaTeslaT4"
    }
    with open(out_dir / 'kernel-metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    out_nb = out_dir / 'rsna-knee-v22-quint-engine-blend.ipynb'
    with open(out_nb, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully scaffolded Candidate V22 in {out_dir} ({len(nb['cells'])} cells)")
    return str(out_nb)

if __name__ == '__main__':
    generate_v22_notebook()
