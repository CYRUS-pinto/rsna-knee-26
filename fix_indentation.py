import json

with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'r') as f:
    nb = json.load(f)

# Fix cell 2 - fix indentation of else block
cell2 = nb['cells'][2]
src = ''.join(cell2['source'])

# Fix the else block indentation
src = src.replace(
    '    else:\n    print(f"[!] {name}: No .pt files found at {path}")',
    '    else:\n        print(f"[!] {name}: No .pt files found at {path}")'
)

cell2['source'] = [src]

with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Fixed indentation error')