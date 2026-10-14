import json

with open('scratch/intel/jenil_956/rsna-knee-0-956-raptor-convnext-r7-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c6 = ''.join(nb['cells'][6]['source'])
for line in c6.splitlines():
    if any(k in line for k in ['assert', 'raise', 'corr', 'rho', 'except']):
        print(line[:120])
