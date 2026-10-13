import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open("scratch/kernels/hitarth/rsna-knee-coatnet-s6-25d-mil-0957.ipynb", encoding='utf-8') as f:
    nb = json.load(f)

for idx in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
    if idx < len(nb['cells']):
        src = "".join(nb['cells'][idx]['source'])
        print(f"=== CELL {idx} ===")
        for line in src.split('\n'):
            if any(k in line.lower() for k in ['input', 'weight', 'slug', 'dataset', 'blend', 'checkpoint', 'models']):
                print("  ", line[:120])
