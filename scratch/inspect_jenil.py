import json

with open('scratch/intel/jenil_956/rsna-knee-0-956-raptor-convnext-r7-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Total cells:', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    first = src.splitlines()[0] if src.splitlines() else ''
    print(f"Cell {i} ({c['cell_type']}): len={len(src)} | {first[:70].encode('ascii', 'replace').decode('ascii')}")
