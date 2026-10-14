import json

with open('scratch/kernels/r7_quint_ensemble/rsna-knee-quint-ensemble-r7-colab.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    first_line = src.splitlines()[0] if src.splitlines() else 'empty'
    print(f"Cell {i} ({c['cell_type']}): {len(src)} chars | {first_line[:80]}")
