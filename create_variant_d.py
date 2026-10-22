import json
import shutil
import os

src = r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb'

# Variant D: Equal MIL weights (Fallback)
dst_d = r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_variant_d_alt_weights\rsna-knee-v50-variant-d.ipynb'
os.makedirs(os.path.dirname(dst_d), exist_ok=True)
shutil.copy2(src, dst_d)

with open(dst_d, 'r') as f:
    nb = json.load(f)

nb['cells'][0]['source'] = ['# RSNA Knee V50 Variant D \u2014 Equal MIL Weights (Fallback)\n', '## Equal weights for all 3 MIL arms']

# Replace the GOLD58_OPTIMIZED_3ENGINE_WEIGHTS with equal weights
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src_text = ''.join(cell['source'])
        if 'GOLD58_OPTIMIZED_3ENGINE_WEIGHTS = {' in src_text:
            new_src = src_text.replace(
                'GOLD58_OPTIMIZED_3ENGINE_WEIGHTS = {',
                'GOLD58_OPTIMIZED_3ENGINE_WEIGHTS = {\n    "ACL": (0.333, 0.333, 0.334),'
            )
            # Replace all the weight entries with equal weights
            for target in ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']:
                new_src = new_src.replace(
                    'GOLD58_OPTIMIZED_3ENGINE_WEIGHTS = {',
                    'GOLD58_OPTIMIZED_3ENGINE_WEIGHTS = {'
                )
            cell['source'] = [new_src]
            print('Modified Variant D weights')
            break

with open(dst_d, 'w') as f:
    json.dump(nb, f, indent=1)
print('Variant D done')