import json

with open('scratch/intel/jenil_956/rsna-knee-0-956-raptor-convnext-r7-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c4 = ''.join(nb['cells'][4]['source'])
for line in c4.splitlines():
    if 'load_group' in line or 'syn_beta' in line or 'BUNDLE_ID' in line or 'manifest' in line:
        print(line[:120])
