import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open("scratch/kernels/apex_v21/rsna-knee-apex-grandmaster-stack-v21.ipynb", encoding='utf-8') as f:
    nb = json.load(f)

for idx in [0, 1, 5, 7]:
    if idx < len(nb.get('cells', [])):
        c = nb['cells'][idx]
        print(f"================== CELL {idx} ({c.get('cell_type')}) ==================")
        print("".join(c.get('source', [])))
