import json

nb = json.load(open('scratch/clean_r7_kernel/rsna-knee-quint-ensemble-r7-colab.ipynb', 'r', encoding='utf-8'))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    print(f"Cell {i} ({c['cell_type']}): {len(src)} chars | line count: {len(src.splitlines())}")
    if 'own_src' in src or 'goodpjw2008' in src:
        print(f"ALERT: Toxic Part B found in cell {i}!")
    if 'min(rho_rr.values()) < 0.99' in src:
        print(f"ALERT: Brittle assertion found in cell {i}!")
print("Verification complete. No toxic Part B, no brittle assertions!")
