import json
import re

for name, path in [
    ("Hitarth 0957", "scratch/kernels/hitarth/rsna-knee-coatnet-s6-25d-mil-0957.ipynb"),
    ("Omniverse SOTA", "scratch/kernels/omniverse/omniverse-sota-rsna-knee-abnormality-detection.ipynb")
]:
    print(f"==================================================")
    print(f"KERNEL: {name}")
    print(f"==================================================")
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    print(f"Total cells: {len(cells)}")
    
    # Look for dataset paths, models, weights, blend ratios
    for idx, c in enumerate(cells):
        src = "".join(c.get('source', []))
        if any(w in src.lower() for w in ['/kaggle/input', 'weight', 'blend', 'ensemble', '0.95', 'auc']):
            lines = [l for l in src.split('\n') if any(w in l.lower() for w in ['/kaggle/input', 'weight', 'blend', 'ensemble', '0.95', 'auc', 'sub'])]
            if lines:
                print(f"--- Cell {idx} ({c.get('cell_type')}) ---")
                for l in lines[:10]:
                    print("  ", l.strip())
