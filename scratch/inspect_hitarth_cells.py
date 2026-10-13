import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open("scratch/kernels/hitarth/rsna-knee-coatnet-s6-25d-mil-0957.ipynb", encoding='utf-8') as f:
    nb = json.load(f)

for idx in range(min(5, len(nb['cells']))):
    print(f"=== HITARTH CELL {idx} ({nb['cells'][idx].get('cell_type')}) ===")
    src = "".join(nb['cells'][idx]['source'])
    print(src[:600])
