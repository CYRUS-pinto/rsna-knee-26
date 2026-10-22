"""
Clone Working Fold-1 Notebooks into Multi-Fold (Folds 2, 3, 4)
================================================================
This script clones the working Fold-1 notebooks (Engine 12 and Engine 13)
into multi-fold notebooks that sequentially run Folds 2, 3, 4.

Rules:
- Keep cells[:5] intact (env setup, auth, fold generation, labels, id2row)
- Clone cells[5:] for Fold 2, 3, 4 with proper replacements
- Replace: "FOLD = 1" -> "FOLD = 2/3/4", "fold1" -> "fold2/3/4", "Fold 1" -> "Fold 2/3/4"
- Append torch.cuda.empty_cache() + gc.collect() at the end of each fold
"""

import json
import re
from pathlib import Path


def clone_notebook(fold1_path, multi_fold_path, engine_num):
    """Clone a working fold1 notebook into a multi-fold notebook."""
    
    with open(fold1_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Keep cells 0-4 (env setup, auth, fold generation, labels, id2row)
    base_cells = nb['cells'][:5]
    
    # Get the training script writing, execution, and publishing cells (cells[3:])
    fold1_cells = nb['cells'][3:]
    
    # Build multi-fold cells
    multi_fold_cells = []
    
    for fold_num in [2, 3, 4]:
        for cell in nb['cells'][3:]:
            new_cell = cell.copy()
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                
                # Replace fold references
                source = source.replace('fold1', f'fold{fold_num}')
                source = source.replace('Fold 1', f'Fold {fold_num}')
                source = source.replace('FOLD = 1', f'FOLD = {fold_num}')
                source = source.replace('FOLD=1', f'FOLD={fold_num}')
                source = source.replace('fold1', f'fold{fold_num}')
                source = source.replace('Fold 1', f'Fold {fold_num}')
                
                # Also handle the specific case of train_engine12_fold1.py -> train_engine12_fold2.py etc
                source = source.replace('train_engine12_fold1.py', f'train_engine12_fold{fold_num}.py')
                source = source.replace('train_engine13_fold1.py', f'train_engine13_fold{fold_num}.py')
                source = source.replace('fold1_best.pt', f'fold{fold_num}_best.pt')
                source = source.replace('fold1_best.pt', f'fold{fold_num}_best.pt')
                source = source.replace('fold1_upload', f'fold{fold_num}_upload')
                source = source.replace('fold1-weights', f'fold{fold_num}-weights')
                source = source.replace('Fold 1', f'Fold {fold_num}')
                source = source.replace('Fold 1', f'Fold {fold_num}')
                
                # Special handling for engine12 vs engine13
                if engine_num == 12:
                    source = source.replace('Engine 12', 'Engine 12')
                    source = source.replace('ConvNeXt-Large', 'ConvNeXt-Large')
                else:
                    source = source.replace('Engine 13', 'Engine 13')
                    source = source.replace('Swin-Base', 'Swin-Base')
                
                new_cell = {
                    'cell_type': 'code',
                    'metadata': cell.get('metadata', {}),
                    'execution_count': None,
                    'outputs': [],
                    'source': [source]
                }
                multi_fold_cells.append(new_cell)
    
    # Add memory cleanup cell after each fold
    cleanup_cell = {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [
            '# Clear GPU memory for next fold\n',
            'import torch, gc\n',
            'torch.cuda.empty_cache()\n',
            'gc.collect()\n',
            'print(f"=== Fold {fold_num} Complete ====", flush=True)\n'
        ]
    }
    multi_fold_cells.append(cleanup_cell)
    
    # Build new notebook
    new_nb = {
        'cells': nb['cells'][:5] + multi_fold_cells,  # Keep first 5 cells + new multi-fold cells
        'metadata': nb['metadata'],
        'nbformat': nb['nbformat'],
        'nbformat_minor': nb['nbformat_minor']
    }
    
    output_path = Path(f'C:\\Users\\Cyrus\\Downloads\\rsna knee fip\\colab_engine{engine_num}_multi_fold_master.ipynb')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'cells': nb['cells'][:5] + multi_fold_cells,
            'metadata': nb['metadata'],
            'nbformat': nb['nbformat'],
            'nbformat_minor': nb['nbformat_minor']
        }, f, indent=1)
    
    print(f'Created: {output_path}')
    return output_path


def main():
    base = Path(r'C:\Users\Cyrus\Downloads\rsna knee fip')
    
    # Task 1: Clone Engine 12
    print('='*60)
    print('TASK 1: Cloning Engine 12 Fold-1 -> Multi-Fold (Folds 2,3,4)')
    print('='*60)
    clone_notebook(
        fold1_path=r'C:\Users\Cyrus\Downloads\rsna knee fip\colab_engine12_fold1_master.ipynb',
        multi_fold_path=Path(r'C:\Users\Cyrus\Downloads\rsna knee fip\colab_engine12_multi_fold_master.ipynb'),
        engine_num=12
    )
    
    # Task 1b: Clone Engine 13
    print('='*60)
    print('TASK 1b: Cloning Engine 13 Fold-1 -> Multi-Fold (Folds 2,3,4)')
    print('='*60)
    clone_notebook(
        fold1_path=r'C:\Users\Cyrus\Downloads\rsna knee fip\colab_engine13_fold1_master.ipynb',
        multi_fold_path=Path(r'C:\Users\Cyrus\Downloads\rsna knee fip\colab_engine13_multi_fold_master.ipynb'),
        engine_num=13
    )
    
    # Verify both files are valid JSON
    print('='*60)
    print('VERIFICATION')
    print('='*60)
    for path in [
        r'C:\Users\Cyrus\Downloads\rsna knee fip\colab_engine12_multi_fold_master.ipynb',
        r'C:\Users\Cyrus\Downloads\rsna knee fip\colab_engine13_multi_fold_master.ipynb'
    ]:
        with open(path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        print(f'{path}: Valid JSON, {len(nb["cells"])} cells')
        # Check multi-fold cell exists
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                src = ''.join(cell['source'])
                if 'for FOLD in [2, 3, 4]:' in src:
                    print(f'  Found multi-fold loop in cell {i}')
                    break
    print('Verification complete.')


if __name__ == '__main__':
    main()