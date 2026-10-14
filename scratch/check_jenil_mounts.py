import json

with open('scratch/intel/jenil_956/rsna-knee-0-956-raptor-convnext-r7-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in [2, 4]:
    src = ''.join(nb['cells'][i]['source'])
    for line in src.splitlines():
        if any(k in line for k in ['find_mount', 'SLUG', 'INPUT /', 'C_SLUG', 'W_SLUG', 'candidate', 'weights']):
            print(f"Cell {i}: {line[:100]}")
