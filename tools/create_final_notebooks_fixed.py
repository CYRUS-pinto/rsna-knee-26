import json
from pathlib import Path

# Read the base notebooks
with open(r'C:\Users\Cyrus\Downloads\colab_engine12_fold1_master.ipynb', 'r', encoding='utf-8') as f:
    engine12_base = json.load(f)

with open(r'C:\Users\Cyrus\Downloads\colab_engine13_fold1_master.ipynb', 'r', encoding='utf-8') as f:
    engine13_base = json.load(f)

# Read the multi-fold source from files
with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\tools\engine12_source.txt', 'r', encoding='utf-8') as f:
    engine12_multi = f.read()

with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\tools\engine13_source.txt', 'r', encoding='utf-8') as f:
    engine13_multi = f.read()

# Create Engine 12 multi-fold notebook
new_cells = engine12_base['cells'][:5]  # Keep first 5 cells
new_cells.append({
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': [engine12_multi]
})
new_cells.append({
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': ['print("=== All Engine 12 Folds Complete ===")']
})

new_nb = {
    'cells': new_cells,
    'metadata': engine12_base['metadata'],
    'nbformat': engine12_base['nbformat'],
    'nbformat_minor': engine12_base['nbformat_minor']
}

with open(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb', 'w', encoding='utf-8') as f:
    json.dump({
        'cells': engine12_base['cells'][:5] + [
            {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [engine12_multi]},
            {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': ['print("=== All Engine 12 Folds Complete ===")']}
        ],
        'metadata': engine12_base['metadata'],
        'nbformat': engine12_base['nbformat'],
        'nbformat_minor': engine12_base['nbformat_minor']
    }, f, indent=1)
print('Created Engine 12 multi-fold')

# Engine 13
with open(r'C:\Users\Cyrus\Downloads\colab_engine13_fold1_master.ipynb', 'r', encoding='utf-8') as f:
    engine13_base = json.load(f)

new_cells = engine13_base['cells'][:5]
new_cells.append({
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': [engine13_multi]
})
new_cells.append({
    'cell_type': 'code',
    'metadata': {},
    'execution_count': None,
    'outputs': [],
    'source': ['print("=== All Engine 13 Folds Complete ===")']
})

new_nb = {
    'cells': engine13_base['cells'][:5] + [
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [engine13_multi]},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': ['print("=== All Engine 13 Folds Complete ===")']}
    ],
    'metadata': engine13_base['metadata'],
    'nbformat': engine13_base['nbformat'],
    'nbformat_minor': engine13_base['nbformat_minor']
}

with open(r'C:\Users\Cyrus\Downloads\colab_engine13_multi_fold_master.ipynb', 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, indent=1)
print('Created Engine 13 multi-fold')