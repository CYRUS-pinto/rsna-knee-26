import json
import re

with open('scratch/kernels/jiweiliu/rsna-knee-fast-2xt4-inference.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

inputs = set()
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    for m in re.findall(r'/kaggle/input/([a-zA-Z0-9_\-\.]+)', src):
        inputs.add(m)

print("Inputs in Jiwei Liu's fast inference notebook:")
for i in sorted(inputs):
    print(" -", i)
