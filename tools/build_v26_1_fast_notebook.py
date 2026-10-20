import json
import os

src_nb = 'submissions/v26_fast_raptor_multiscale/rsna-knee-fast-raptor-multiscale.ipynb'
dst_dir = 'submissions/v26_1_fast_efficiency_master'
os.makedirs(dst_dir, exist_ok=True)

with open(src_nb, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0: Header markdown
md_header = """# RSNA Knee - Candidate V26.1 Fast MultiScale Raptor Efficiency Master
#
# PURPOSE: $18,000 Efficiency Prize Track Champion.
# Evaluates Multi-Scale Raptor (384px + 224px SWA) with 75/25 capacity-weighted rank fusion.
# Target runtime on hidden test set: ~5.5 minutes (~330s).
# Target metric: Minimize Efficiency Score while preserving AUC >= 0.950.
"""
nb['cells'][0]['source'] = [md_header]

# Cell 5: Calibrate A224 to 0.25 (75% 384px, 25% 224px)
c5 = "".join(nb['cells'][5]['source'])
target_line = "A224 = 0.35  # Verified optimal multiscale split: 65% 384px, 35% 224px"
replacement_line = "A224 = 0.25  # Calibrated capacity split: 75% 384px (0.949 standalone), 25% 224px"
assert target_line in c5, f"target_line not found in cell 5"
c5 = c5.replace(target_line, replacement_line)

target_desc = "'architecture': 'Raptor-MultiScale-Fast (384 + 224)'"
replacement_desc = "'architecture': 'Candidate V26.1 Fast MultiScale Raptor (75/25 Split)'"
c5 = c5.replace(target_desc, replacement_desc)
nb['cells'][5]['source'] = [c5]

# Write out notebook
dst_nb = os.path.join(dst_dir, 'rsna-knee-v26-1-fast-efficiency-master.ipynb')
with open(dst_nb, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

# Write kernel metadata
meta = {
  "id": "cyruspinto/rsna-knee-v26-1-fast-efficiency-master",
  "title": "RSNA Knee V26.1 Fast MultiScale Raptor Efficiency Master",
  "code_file": "rsna-knee-v26-1-fast-efficiency-master.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "keywords": ["gpu"],
  "dataset_sources": [
    "nartaa/rsna-knee-publication-swa-weights-20261007"
  ],
  "kernel_sources": [],
  "competition_sources": ["rsna-knee-abnormality-detection"],
  "model_sources": [],
  "machine_shape": "NvidiaTeslaT4"
}

with open(os.path.join(dst_dir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

print("[+] Successfully generated Candidate V26.1 notebook and kernel metadata in", dst_dir)
