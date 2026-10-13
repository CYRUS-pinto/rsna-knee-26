import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open('scratch/kernels/steven_cache/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, c in enumerate(nb['cells']):
    src = "".join(c.get('source', []))
    if any(k in src.lower() for k in ['dataset', 'model', 'train', 'cache', 'dataloader', 'epoch', 'auc']):
        print(f"=== Cell {idx} ({c.get('cell_type')}) ===")
        for l in src.split('\n')[:15]:
            print("  ", l[:110])
