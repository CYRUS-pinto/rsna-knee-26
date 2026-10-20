import json
import os
from pathlib import Path

src_nb = 'submissions/v24_r7_uniform_multiscale_master/rsna-knee-v24-r7-uniform-multiscale-master.ipynb'
dst_dir = 'submissions/v25_r7_uniform_multiscale_fliptta'
os.makedirs(dst_dir, exist_ok=True)

with open(src_nb, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 6: Enable FLIP_TTA = True
c6 = "".join(nb['cells'][6]['source'])
target_line = "FLIP_TTA = False   # [d957] package F: no flip TTA"
replacement_line = "FLIP_TTA = True    # [v25] Enable horizontal flip TTA on ConvNeXt R7 for bilateral symmetry averaging"

assert target_line in c6, "target_line not found in cell 6"
c6 = c6.replace(target_line, replacement_line)
nb['cells'][6]['source'] = [c6]

# Save notebook
dst_nb = os.path.join(dst_dir, 'rsna-knee-v25-r7-uniform-multiscale-fliptta.ipynb')
with open(dst_nb, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

# Save kernel-metadata.json
meta = {
  "id": "cyruspinto/rsna-knee-v25-r7-uniform-multiscale-fliptta",
  "title": "RSNA Knee V25 R7 Uniform MultiScale FlipTTA",
  "code_file": "rsna-knee-v25-r7-uniform-multiscale-fliptta.ipynb",
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
    "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"
  ],
  "kernel_sources": [],
  "competition_sources": ["rsna-knee-abnormality-detection"],
  "model_sources": [],
  "machine_shape": "NvidiaTeslaT4"
}

with open(os.path.join(dst_dir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

print("[+] Successfully generated Candidate V25 notebook and kernel metadata in", dst_dir)
