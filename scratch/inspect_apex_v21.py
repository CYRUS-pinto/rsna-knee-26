import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("scratch/kernels/apex_v21/rsna-knee-apex-grandmaster-stack-v21.ipynb", encoding='utf-8') as f:
    nb = json.load(f)

inputs = set()
for c in nb.get('cells', []):
    src = "".join(c.get('source', []))
    matches = re.findall(r'/kaggle/input/([a-zA-Z0-9_\-\.]+)', src)
    for m in matches:
        inputs.add(m)

print("Unique datasets / inputs referenced in Apex Grandmaster Stack V21:")
for i in sorted(inputs):
    print(" -", i)

print("\nMarkdown and strategy cells in V21:")
for idx, c in enumerate(nb.get('cells', [])):
    src = "".join(c.get('source', []))
    if c.get('cell_type') == 'markdown' or any(k in src.lower() for k in ['v21', 'v20', 'blend', 'weights', '0.95']):
        first_line = src.split('\n')[0] if src else ""
        print(f"Cell {idx}: {first_line[:80]}")
