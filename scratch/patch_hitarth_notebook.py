import json

path = 'scratch/kernels/hitarth/rsna-knee-coatnet-s6-25d-mil-0957.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

cell_src = "".join(nb['cells'][6]['source'])
old_code = "M_SRC = find_mount(C_SLUG, 'manifest.json')"
new_code = """M_SRC = None
for cand_c in ['rsna-knee-compact-v1', 'rsna-knee-compact-r6cnxfsss', 'compact_models']:
    try:
        M_SRC = find_mount(cand_c, 'manifest.json')
        C_SLUG = cand_c
        break
    except Exception:
        pass
if M_SRC is None:
    M_SRC = find_mount(C_SLUG, 'manifest.json')"""

if old_code in cell_src:
    nb['cells'][6]['source'] = [cell_src.replace(old_code, new_code)]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully patched Cell 6 in Hitarth notebook!")
else:
    print("Warning: old_code not found in Cell 6")
