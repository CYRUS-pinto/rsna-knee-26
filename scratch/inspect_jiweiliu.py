import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open('scratch/kernels/jiweiliu/rsna-knee-fast-2xt4-inference.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in Jiwei Liu notebook: {len(nb['cells'])}")
for idx, c in enumerate(nb['cells'][:10]):
    src = "".join(c.get('source', []))
    print(f"=== Cell {idx} ({c.get('cell_type')}) ===")
    for l in src.split('\n')[:8]:
        print("  ", l[:100])
