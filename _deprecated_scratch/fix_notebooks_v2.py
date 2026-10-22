import json
from pathlib import Path

def fix_notebook(engine_num):
    if engine_num == 12:
        src = Path(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb')
        dst = Path(r'C:\Users\Cyrus\Downloads\colab_engine12_multi_fold_master.ipynb')
    else:
        src = Path(r'C:\Users\Cyrus\Downloads\colab_engine13_multi_fold_master.ipynb')
        dst = Path(r'C:\Users\Cyrus\Downloads\colab_engine13_multi_fold_master.ipynb')

    with open(src, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the multi-fold loop cell (cell 2)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src_text = ''.join(cell['source'])
            if 'for FOLD in [2, 3, 4]:' in src_text:
                print(f'Found multi-fold cell at index {i}')
                # Read the fixed source from file
                if engine_num == 12:
                    with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\tools\engine12_fixed_source.txt', 'r', encoding='utf-8') as f:
                        fixed_source = f.read()
                else:
                    with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\tools\engine13_fixed_source.txt', 'r', encoding='utf-8') as f:
                        fixed_source = f.read()
                nb['cells'][i]['source'] = [fixed_source]
                print(f'Fixed cell {i}')
                break

    with open(dst, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'Fixed notebook saved to {dst}')

if __name__ == '__main__':
    fix_notebook(12)
    fix_notebook(13)
    print('Done!')