import json

nb_path = 'submissions/v32_tri_engine_medical_master/rsna-knee-v32-tri-engine-medical-master.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

# Patch Cell 8 (rad_run.py)
c8_src = ''.join(nb['cells'][8]['source'])
old_c8 = '''# Read metadata and match slots
test_series = pd.read_csv(DATA / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
desc = test_series["SeriesDescription"].fillna("").str.lower()
fs_opts = test_series["ScanOptions"].fillna("").str.upper()
is_fs = desc.str.contains("fs|fat|fatsat") | fs_opts.str.contains("FAT|SAT")
test_series["fatsat"] = is_fs'''

new_c8 = '''# Read metadata and match slots
test_series = pd.read_csv(DATA / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
if "Fat_Suppression" in test_series.columns:
    test_series["fatsat"] = pd.to_numeric(test_series["Fat_Suppression"], errors="coerce").fillna(0).astype(int) == 1
else:
    desc = test_series.get("SeriesDescription", pd.Series("", index=test_series.index)).fillna("").str.lower()
    fs_opts = test_series.get("ScanOptions", pd.Series("", index=test_series.index)).fillna("").str.upper()
    test_series["fatsat"] = desc.str.contains("fs|fat|fatsat") | fs_opts.str.contains("FAT|SAT")'''

assert old_c8 in c8_src, "Could not find old_c8 in cell 8"
c8_src = c8_src.replace(old_c8, new_c8)
nb['cells'][8]['source'] = [c8_src]

# Patch Cell 10 (Linear rank fusion + zero-variance assertion)
c10_src = ''.join(nb['cells'][10]['source'])
old_blend = '''    # Primary Blend (Raptor + ConvNeXt)
    wr = W_RAPTOR_C.get(c, 0.50)
    rr_safe = np.clip(np.nan_to_num(rr, nan=0.5), 0.0, 1.0)
    ro_safe = np.clip(np.nan_to_num(ro, nan=0.5), 0.0, 1.0)
    primary = (wr * (rr_safe ** p_power) + (1.0 - wr) * (ro_safe ** p_power)) ** (1.0 / p_power)
    primary[rap_fail] = ro_safe[rap_fail]
    primary[our_fail] = rr_safe[our_fail]'''

new_blend = '''    # Primary Blend (Raptor + ConvNeXt) - Robust Linear Rank Blend
    wr = 0.50  # Balanced 50/50 prior-free rank fusion
    rr_safe = np.clip(np.nan_to_num(rr, nan=0.5), 0.0, 1.0)
    ro_safe = np.clip(np.nan_to_num(ro, nan=0.5), 0.0, 1.0)
    primary = wr * rr_safe + (1.0 - wr) * ro_safe
    primary[rap_fail] = ro_safe[rap_fail]
    primary[our_fail] = rr_safe[our_fail]'''

assert old_blend in c10_src, "Could not find old_blend in cell 10"
c10_src = c10_src.replace(old_blend, new_blend)

old_assert = 'assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()'
new_assert = '''assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
if rad_exists:
    rad_std = rad[labels].values.std()
    print(f"RadImageNet raw std: {rad_std:.4f}", flush=True)
    assert rad_std > 0.001, "RadImageNet predictions collapsed to uniform fallback constants!"'''

assert old_assert in c10_src, "Could not find old_assert in cell 10"
c10_src = c10_src.replace(old_assert, new_assert)
nb['cells'][10]['source'] = [c10_src]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Successfully patched V32 notebook!")
