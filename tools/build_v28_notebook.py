import json
import os

src_nb = 'submissions/v25_r7_uniform_multiscale_fliptta/rsna-knee-v25-r7-uniform-multiscale-fliptta.ipynb'
dst_dir = 'submissions/v28_calibrated_grandmaster_master'
os.makedirs(dst_dir, exist_ok=True)

with open(src_nb, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 8: Fix Stage 3 blending weights
c8 = "".join(nb['cells'][8]['source'])

old_weights_block = """W_RAPTOR_C = {
    'ACL': 0.50,
    'MCL': 0.50,
    'Medial Meniscus': 0.50,
    'Lateral Meniscus': 0.50,
    'Medial OA': 0.50,
    'Lateral OA': 0.50,
    'PF OA': 0.50,
    'Effusion': 0.50,
    'Synovitis': 0.50,
    "Baker's": 0.50,
    'Contusion': 0.50,
    'Fracture': 0.50
}"""

new_weights_block = """# [v28] Restored & Calibrated Specialist Weights (Fixing V24 uniform regression from 0.958 -> 0.956)
# Leverages architectural inductive bias:
# - Raptor specializes in whole-knee attention (Synovitis 0.80, ACL 0.60, PF OA 0.60)
# - ConvNeXt R7 specializes in joint-space narrowing & fluid stackers (Lateral OA 0.20 Raptor, Baker's 0.35 Raptor)
# - Balanced compartmental stability on Menisci & Medial OA (0.50 / 0.50)
W_RAPTOR_C = {
    'Lateral OA': 0.20,
    'Synovitis': 0.80,
    'PF OA': 0.60,
    "Baker's": 0.35,
    'ACL': 0.60,
    'MCL': 0.50,
    'Medial Meniscus': 0.50,
    'Lateral Meniscus': 0.50,
    'Medial OA': 0.50,
    'Effusion': 0.50,
    'Contusion': 0.50,
    'Fracture': 0.50
}"""

assert old_weights_block in c8, "old_weights_block not found in cell 8"
c8 = c8.replace(old_weights_block, new_weights_block)

# Also ensure A224 is calibrated to 0.30
c8 = c8.replace("A224 = 0.35", "A224 = 0.30")

target_desc = "'mode': 'blend'"
replacement_desc = "'mode': 'Candidate V28 Calibrated Grandmaster Master (Specialist Induction + ConvNeXt Flip TTA)'"
c8 = c8.replace(target_desc, replacement_desc)

nb['cells'][8]['source'] = [c8]

# Save notebook
dst_nb = os.path.join(dst_dir, 'rsna-knee-v28-calibrated-grandmaster-master.ipynb')
with open(dst_nb, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

# Save kernel-metadata.json
meta = {
  "id": "cyruspinto/rsna-knee-v28-calibrated-grandmaster-master",
  "title": "RSNA Knee V28 Calibrated Grandmaster Master",
  "code_file": "rsna-knee-v28-calibrated-grandmaster-master.ipynb",
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

print("[+] Successfully generated Candidate V28 notebook and kernel metadata in", dst_dir)
