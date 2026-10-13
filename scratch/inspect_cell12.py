import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open("scratch/kernels/apex_v21/rsna-knee-apex-grandmaster-stack-v21.ipynb", encoding='utf-8') as f:
    nb = json.load(f)

for idx in [1, 2]:
    print(f"=== CELL {idx} ===")
    print("".join(nb['cells'][idx]['source']))
