import json
import os

src_nb = 'submissions/v25_r7_uniform_multiscale_fliptta/rsna-knee-v25-r7-uniform-multiscale-fliptta.ipynb'
dst_dir = 'submissions/v27_r7_multiscale_capacity_master'
os.makedirs(dst_dir, exist_ok=True)

with open(src_nb, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 8: In Stage 3, update A224 to 0.25
c8 = "".join(nb['cells'][8]['source'])
target_line = "A224 = 0.35   # [d957] share of the Raptor half given to the 224-crop model"
replacement_line = "A224 = 0.25   # [v27] Calibrated capacity split: 75% 384px (0.949 standalone), 25% 224px"
assert target_line in c8, f"target_line not found in cell 8"
c8 = c8.replace(target_line, replacement_line)

target_desc = "'mode': 'blend'"
replacement_desc = "'mode': 'Candidate V27 R7 MultiScale Capacity Master (75/25 Raptor + ConvNeXt Flip TTA)'"
c8 = c8.replace(target_desc, replacement_desc)
nb['cells'][8]['source'] = [c8]

# Save notebook
dst_nb = os.path.join(dst_dir, 'rsna-knee-v27-r7-multiscale-capacity-master.ipynb')
with open(dst_nb, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

# Save kernel-metadata.json
meta = {
  "id": "cyruspinto/rsna-knee-v27-r7-multiscale-capacity-master",
  "title": "RSNA Knee V27 R7 MultiScale Capacity Master",
  "code_file": "rsna-knee-v27-r7-multiscale-capacity-master.ipynb",
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

print("[+] Successfully generated Candidate V27 notebook and kernel metadata in", dst_dir)
