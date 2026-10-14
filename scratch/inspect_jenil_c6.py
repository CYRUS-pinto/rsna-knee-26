import json

with open('scratch/intel/jenil_956/rsna-knee-0-956-raptor-convnext-r7-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c6 = ''.join(nb['cells'][6]['source'])
print(c6[:2000])
