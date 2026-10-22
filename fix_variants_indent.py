import json
import os

variants = [
    r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_variant_a_low_anchor\rsna-knee-v50-variant-a.ipynb',
    r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_variant_b_high_anchor\rsna-knee-v50-variant-b.ipynb',
    r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_variant_c_no_raptor\rsna-knee-v50-variant-c.ipynb',
    r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_variant_d_alt_weights\rsna-knee-v50-variant-d.ipynb',
]

for path in variants:
    with open(path, 'r') as f:
        nb = json.load(f)

    if len(nb['cells']) > 2:
        cell2 = nb['cells'][2]
        src = ''.join(cell2['source'])
        src = src.replace(
            '    else:\n    print(f"[!] {name}: No .pt files found at {path}")',
            '    else:\n        print(f"[!] {name}: No .pt files found at {path}")'
        )
        nb['cells'][2]['source'] = [src]

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'Fixed {path}')

print('All variants fixed!')