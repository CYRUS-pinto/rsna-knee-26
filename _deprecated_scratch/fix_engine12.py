import json
from pathlib import Path

# Read Engine 12 multi-fold notebook
src = Path(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb')
with open(src, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Read the fixed source
with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\tools\engine12_source.txt', 'r', encoding='utf-8') as f:
    multi_fold_source = f.read()

# Find the multi-fold cell (index 2)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src_text = ''.join(cell['source'])
        if 'for FOLD in [2, 3, 4]:' in src_text:
            nb['cells'][i]['source'] = [open(r'C:\Users\Cyrus\Downloads\rsna knee fip\tools\engine12_source.txt', 'r', encoding='utf-8').read()]
            print(f'Fixed Engine 12 cell {i}')
            break

# Save
with open(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Fixed Engine 12')