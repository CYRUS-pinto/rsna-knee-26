import json
from pathlib import Path

nb_path = 'submissions/v32_tri_engine_medical_master/rsna-knee-v32-tri-engine-medical-master.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

# Patch Cell 8 (rad_run.py) path discovery
c8_src = ''.join(nb['cells'][8]['source'])

old_data_lookup = '''DATA = Path("/kaggle/input/rsna-knee-abnormality-detection")
if not DATA.exists():
    for p in Path("/kaggle/input").glob("*/test.csv"):
        DATA = p.parent
        break'''

new_data_lookup = '''DATA = None
for cand in [
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
]:
    if (cand / "test_series.csv").is_file():
        DATA = cand
        break
if DATA is None:
    for p in Path("/kaggle/input").rglob("test_series.csv"):
        DATA = p.parent
        break
assert DATA is not None and (DATA / "test_series.csv").is_file(), f"Competition test data not found! Search checked: {list(Path('/kaggle/input').glob('*'))}"'''

if old_data_lookup in c8_src:
    c8_src = c8_src.replace(old_data_lookup, new_data_lookup)
else:
    # Look for DATA = in c8_src
    idx = c8_src.find('DATA = Path(')
    end_idx = c8_src.find('print(f"Loaded', idx)
    target = c8_src[idx:end_idx]
    c8_src = c8_src.replace(target, new_data_lookup + '\n\n')

nb['cells'][8]['source'] = [c8_src]

# Patch Cell 9: Enforce hard failure if rad_run exits with non-zero
c9_src = ''.join(nb['cells'][9]['source'])
old_c9 = '''if ret.returncode != 0:
    print("WARNING: RadImageNet E11 exited with error:", ret.stderr)'''
new_c9 = '''if ret.returncode != 0:
    print("ERROR: RadImageNet E11 failed with stderr:", ret.stderr)
    raise RuntimeError(f"RadImageNet engine crashed with code {ret.returncode}: {ret.stderr}")'''
if old_c9 in c9_src:
    c9_src = c9_src.replace(old_c9, new_c9)
    nb['cells'][9]['source'] = [c9_src]

# Patch Cell 10: Robust fallback creation and dynamic path lookup
c10_src = ''.join(nb['cells'][10]['source'])
old_c10_data = 'DATA = Path("/kaggle/input/rsna-knee-abnormality-detection")'
new_c10_data = '''DATA = None
for cand in [
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
]:
    if (cand / "test_series.csv").is_file():
        DATA = cand
        break
if DATA is None:
    for p in Path("/kaggle/input").rglob("test_series.csv"):
        DATA = p.parent
        break'''
if old_c10_data in c10_src:
    c10_src = c10_src.replace(old_c10_data, new_c10_data)

# Fix dummy rad dataframe creation in cell 10
old_rad_dummy = "rad = pd.read_csv(rad_raw_path, dtype={'StudyInstanceUID': str}) if rad_exists else pd.DataFrame(0.5, index=range(len(test_ids)), columns=['StudyInstanceUID'] + labels)"
new_rad_dummy = '''if rad_exists:
    rad = pd.read_csv(rad_raw_path, dtype={'StudyInstanceUID': str})
else:
    rad = pd.DataFrame(0.5, index=range(len(test_ids)), columns=['StudyInstanceUID'] + labels)
    rad['StudyInstanceUID'] = test_ids'''
if old_rad_dummy in c10_src:
    c10_src = c10_src.replace(old_rad_dummy, new_rad_dummy)

nb['cells'][10]['source'] = [c10_src]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully updated V32 with robust Kaggle competition mount path and strict failure gates!")
