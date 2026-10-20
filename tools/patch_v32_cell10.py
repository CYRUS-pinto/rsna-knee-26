import json

nb_path = 'submissions/v32_tri_engine_medical_master/rsna-knee-v32-tri-engine-medical-master.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))
c10_src = ''.join(nb['cells'][10]['source'])

old_assert = '''if rad_exists:
    rad_std = rad[labels].values.std()
    print(f"RadImageNet raw std: {rad_std:.4f}", flush=True)
    assert rad_std > 0.001, "RadImageNet predictions collapsed to uniform fallback constants!"'''

new_assert = '''assert rad_exists, "CRITICAL: rad_run.py failed to execute or produce rad_raw.csv!"
rad_std = rad[labels].values.std()
print(f"RadImageNet raw std: {rad_std:.4f}", flush=True)
assert rad_std > 0.001, "RadImageNet predictions collapsed to uniform fallback constants!"'''

assert old_assert in c10_src, "old_assert not found in cell 10"
c10_src = c10_src.replace(old_assert, new_assert)
nb['cells'][10]['source'] = [c10_src]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Successfully hardened Cell 10 assertions in V32!")
