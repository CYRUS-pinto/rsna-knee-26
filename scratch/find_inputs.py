import json
import re

with open("scratch/kernels/hitarth/rsna-knee-coatnet-s6-25d-mil-0957.ipynb", encoding='utf-8') as f:
    nb = json.load(f)

inputs = set()
for c in nb.get('cells', []):
    src = "".join(c.get('source', []))
    matches = re.findall(r'/kaggle/input/([a-zA-Z0-9_\-\.]+)', src)
    for m in matches:
        inputs.add(m)

print("Unique datasets / inputs referenced in Hitarth 0957:")
for i in sorted(inputs):
    print(" -", i)
