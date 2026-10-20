import json, os, shutil
from pathlib import Path

src_dir = Path('submissions/v32_tri_engine_medical_master')
dst_dir = Path('submissions/v32_ablation_stride2_norad')
dst_dir.mkdir(parents=True, exist_ok=True)

# 1. Update kernel-metadata.json
meta_path = src_dir / 'kernel-metadata.json'
meta = json.loads(meta_path.read_text(encoding='utf-8'))
meta['id'] = 'cyruspinto/rsna-knee-v32-ablation-stride2-norad'
meta['title'] = 'RSNA Knee V32 Ablation Stride2 NoRad Master'
meta['code_file'] = 'rsna-knee-v32-ablation-stride2-norad.ipynb'
(dst_dir / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

# 2. Modify notebook
nb_path = src_dir / 'rsna-knee-v32-tri-engine-medical-master.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

# In Cell 10, set all W_RAD to 0.0
c10_src = ''.join(nb['cells'][10]['source'])
old_w_rad = '''W_RAD = {
    'Lateral Meniscus': 0.35,
    'Synovitis': 0.30,
    'Contusion': 0.30,
    'Lateral OA': 0.25,
    'MCL': 0.20,
    'ACL': 0.20,
    'Medial OA': 0.20,
    'PF OA': 0.20,
    'Medial Meniscus': 0.20,
    'Effusion': 0.15,
    "Baker's": 0.00,
    'Fracture': 0.00
}'''

new_w_rad = '''# Ablation Control: All RadImageNet weights set strictly to 0.0
W_RAD = {c: 0.0 for c in [
    'ACL', "Baker's", 'Contusion', 'Effusion', 'Fracture',
    'Lateral Meniscus', 'Lateral OA', 'MCL',
    'Medial Meniscus', 'Medial OA', 'PF OA', 'Synovitis'
]}'''

assert old_w_rad in c10_src, "Could not find W_RAD in cell 10"
c10_src = c10_src.replace(old_w_rad, new_w_rad)
nb['cells'][10]['source'] = [c10_src]

# Save ablation notebook
dst_nb = dst_dir / 'rsna-knee-v32-ablation-stride2-norad.ipynb'
with open(dst_nb, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully created V32 Stride-2 Pure Ablation notebook at:", dst_nb)
