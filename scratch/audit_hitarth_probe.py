import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open('scratch/kernels/hitarth/rsna-knee-coatnet-s6-25d-mil-0957.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in Hitarth notebook: {len(nb['cells'])}")
for idx in [6, 7, 8, 9, 14, 15]:
    if idx < len(nb['cells']):
        print(f"\n==================== CELL {idx} ({nb['cells'][idx].get('cell_type')}) ====================")
        src = "".join(nb['cells'][idx].get('source', []))
        for line in src.split('\n'):
            if any(k in line.lower() for k in ['probe', 'constant', '0.5', 'submission', 'ours_sub', 'raptor_sub', 'target_cols', 'labels']):
                print("  ", line[:120])
