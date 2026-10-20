import json

nb_path = 'submissions/v32_tri_engine_medical_master/rsna-knee-v32-tri-engine-medical-master.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))
c8_src = ''.join(nb['cells'][8]['source'])

target = '''    for slot_idx, (sname, plane, req_fs) in enumerate(E11_SLOTS):
        cand = g[(g["Anatomical_Plane"] == plane) & (g["fatsat"] == req_fs)]
        if cand.empty:
            continue
        ser_uid = cand.iloc[0]["SeriesInstanceUID"]'''

replacement = '''    for slot_idx, (sname, plane, req_fs) in enumerate(E11_SLOTS):
        cand = g[(g["Anatomical_Plane"] == plane) & (g["fatsat"] == req_fs)]
        if cand.empty:
            continue
        if len(cand) > 1:
            slice_counts = []
            for s_uid in cand["SeriesInstanceUID"]:
                p = DATA / "test_series" / uid / s_uid
                slice_counts.append(len(os.listdir(p)) if p.is_dir() else 0)
            cand = cand.iloc[np.argmax(slice_counts):np.argmax(slice_counts)+1]
        ser_uid = cand.iloc[0]["SeriesInstanceUID"]'''

assert target in c8_src, "Target string not found in cell 8"
c8_src = c8_src.replace(target, replacement)
nb['cells'][8]['source'] = [c8_src]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Successfully updated V32 cell 8 with deterministic slice-count tie-breaking!")
