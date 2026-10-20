import json, ast
from pathlib import Path
import numpy as np
import pandas as pd

src_nb = json.load(open('submissions/v32_tri_engine_medical_master/rsna-knee-v32-tri-engine-medical-master.ipynb', encoding='utf-8'))
dst_path = Path('submissions/v32_ablation_stride2_norad/rsna-knee-v32-ablation-stride2-norad.ipynb')

c10_src = ''.join(src_nb['cells'][10]['source'])

# Properly handle the assertion in ablation: if rad_exists, indent the checks, or just pass
old_assert_block = '''assert rad_exists, "CRITICAL: rad_run.py failed to execute or produce rad_raw.csv!"
rad_std = rad[labels].values.std()
print(f"RadImageNet raw std: {rad_std:.4f}", flush=True)
assert rad_std > 0.001, "RadImageNet predictions collapsed to uniform fallback constants!"'''

new_assert_block = '''# Ablation control: all W_RAD = 0.0, rad_std assertion is non-fatal warning if rad executed
if rad_exists:
    rad_std = rad[labels].values.std()
    print(f"RadImageNet raw std (ablation control): {rad_std:.4f}", flush=True)
else:
    print("Ablation control: rad_raw.csv not present (expected when rad arm bypassed).", flush=True)'''

assert old_assert_block in c10_src, "old_assert_block not found in cell 10"
c10_src = c10_src.replace(old_assert_block, new_assert_block)
c10_src = c10_src.replace("'mode': 'Candidate V32 Tri-Engine Non-FS Medical Master (Raptor + R7 + RadImageNet E11)'", "'mode': 'Candidate V32-Ablation Stride-2 Control Twin (Raptor + R7 Stride-2, W_RAD=0)'")

old_w_rad = """W_RAD = {
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
}"""

new_w_rad = """# Ablation Control: All RadImageNet weights set strictly to 0.0
W_RAD = {c: 0.0 for c in [
    'Lateral Meniscus', 'Synovitis', 'Contusion', 'Lateral OA',
    'MCL', 'ACL', 'Medial OA', 'PF OA', 'Medial Meniscus',
    'Effusion', "Baker's", 'Fracture'
]}"""

assert old_w_rad in c10_src, "old_w_rad not found in cell 10"
c10_src = c10_src.replace(old_w_rad, new_w_rad)
src_nb['cells'][10]['source'] = [c10_src]

# Validate with ast.parse
clean_lines = []
for line in c10_src.splitlines():
    if line.strip().startswith('%') or line.strip().startswith('!'):
        clean_lines.append('# ' + line)
    else:
        clean_lines.append(line)
ast.parse('\n'.join(clean_lines))

with open(dst_path, 'w', encoding='utf-8') as f:
    json.dump(src_nb, f, indent=1)
print("Successfully generated and AST-validated ablation twin from V32!")
