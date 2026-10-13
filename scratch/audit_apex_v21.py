import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
with open('scratch/kernels/apex_v21/rsna-knee-apex-grandmaster-stack-v21.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for idx, c in enumerate(nb['cells']):
    src = "".join(c.get('source', []))
    if any(k in src.lower() for k in ['required_files', 'find_checkpoint', 'find_mount', 'submission.csv', 'radimagenet']):
        print(f"=== Cell {idx} ({c.get('cell_type')}) ===")
        for l in src.split('\n'):
            if any(k in l.lower() for k in ['def find', 'required_files', 'slug', 'c_slug', 'blend', 'out_path', 'rad_weights']):
                print("  ", l[:120])
