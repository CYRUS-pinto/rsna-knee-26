import json
from pathlib import Path

def read_source(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def create_notebook(engine_num):
    if engine_num == 12:
        src = Path(r'C:\Users\Cyrus\Downloads\colab_engine12_fold1_master.ipynb')
        dst = Path(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb')
        src_path = Path(r'C:\Users\Cyrus\Downloads\rsna knee fip\tools\engine12_source.txt')
    else:
        src = Path(r'C:\Users\Cyrus\Downloads\colab_engine13_fold1_master.ipynb')
        dst = Path(r'C:\Users\Cyrus\Downloads\colab_engine13_multi_fold_master.ipynb')
        src_path = Path(r'C:\Users\Cyrus\Downloads\rsna knee fip\tools\engine13_source.txt')
    
    with open(src) as f:
        nb = json.load(f)
    
    # Keep cells 0 (env) and 1 (auth/download)
    new_cells = [nb['cells'][0], nb['cells'][1]]
    
    # Read source content
    with open(src_path, 'r', encoding='utf-8') as f:
        src_content = f.read()
    
    # Multi-fold loop cell
    new_cells.append({
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [src_content]
    })
    
    # Final completion cell
    new_cells.append({
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': ['print("=== All Folds Complete ===")']
    })
    
    # Create new notebook
    new_nb = {
        'cells': new_cells,
        'metadata': nb['metadata'],
        'nbformat': nb['nbformat'],
        'nbformat_minor': nb['nbformat_minor']
    }
    
    if engine_num == 12:
        dst = Path(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb')
    else:
        dst = Path(r'C:\Users\Cyrus\Downloads\colab_engine13_multi_fold_master.ipynb')
    
    new_nb = {
        'cells': new_cells,
        'metadata': nb['metadata'],
        'nbformat': nb['nbformat'],
        'nbformat_minor': nb['nbformat_minor']
    }
    
    with open(dst, 'w') as f:
        json.dump({'cells': new_cells, 'metadata': nb['metadata'], 'nbformat': nb['nbformat'], 'nbformat_minor': nb['nbformat_minor']}, f, indent=1)
    
    print(f'Created: {dst}')

if __name__ == '__main__':
    create_notebook(12)
    create_notebook(13)
    print('Done!')