import json, os, glob

nb_path = 'scratch/kernels/hitarth/rsna-knee-coatnet-s6-25d-mil-0957.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell3_src = "".join(nb['cells'][3]['source'])

# Target find_weight_file block to replace
old_fn_start = "def find_weight_file(fname):"
idx_start = cell3_src.find(old_fn_start)
assert idx_start != -1, "Could not find old find_weight_file start"

idx_end = cell3_src.find("def main():", idx_start)
assert idx_end != -1, "Could not find def main() boundary"

new_block = '''def find_weight_file(fname):
    # direct dataset mounts first; NEVER recursive-glob the competitions DICOM tree.
    direct = [
        f"/kaggle/input/raptor-knee-arms/{fname}",
        f"/kaggle/input/raptor-knee-arms/1/{fname}",
        f"/kaggle/input/raptor-cnn336/{fname}",
        f"/kaggle/input/rsna-knee-colab-super-weights/{fname}",
        f"/kaggle/input/rsna-knee-colab-super-weights/1/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-colab-super-weights/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-colab-super-weights/1/{fname}",
        f"/kaggle/input/datasets/cyruspinto/rsna-knee-colab-super-weights/versions/1/{fname}",
    ]
    for p in direct:
        if os.path.exists(p):
            return p
    for root, dirs, files in os.walk("/kaggle/input"):
        # skip competition DICOM data to avoid hanging
        if any(bad in root.lower() for bad in ("test_series", "train_images", "train_series", "competitions", "abnormality-detection")):
            dirs.clear()
            continue
        if fname in files:
            return os.path.join(root, fname)
    raise RuntimeError(f"{fname} not found under /kaggle/input")


CANDIDATE_ARMS = [
    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.40},
    {"file": "colab_super_coatnet_0917_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.35},
    {"file": "colab_super_convnext_0917_swa.pt", "arch": "convnext_small.fb_in22k_ft_in1k_384", "res": 384, "w": 0.25},
]
ARMS = []
for a in CANDIDATE_ARMS:
    try:
        p = find_weight_file(a["file"])
        ARMS.append(a)
        print(f"[ARMS] Found weight for {a['file']} at: {p}")
    except Exception as e:
        print(f"[ARMS] Could not find {a['file']}: {e}")
if not ARMS:
    ARMS = [CANDIDATE_ARMS[0]]
print(f"[ARMS] Active model arms ({len(ARMS)}): {[a['file'] for a in ARMS]} with weights {[a.get('w', 1.0) for a in ARMS]}")


'''

cell3_src = cell3_src[:idx_start] + new_block + cell3_src[idx_end:]
nb['cells'][3]['source'] = [cell3_src]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully replaced find_weight_file and CANDIDATE_ARMS block!")
