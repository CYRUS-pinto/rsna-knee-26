import json

with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'r') as f:
    nb = json.load(f)

# Fix cell 3 (index 3) - update corpus loading paths
cell3 = nb['cells'][3]
src3 = ''.join(cell3['source'])

# Update all the hardcoded /content/corpus/ paths to use corpus_path variables
src3 = src3.replace(
    "va = np.load('/content/corpus/all_vols.npy', mmap_mode='r')",
    "va = np.load(os.path.join(corpus_path, 'all_vols.npy'), mmap_mode='r')"
)
src3 = src3.replace(
    "vb = np.load('/content/corpus/extra_vols.npy', mmap_mode='r') if os.path.exists('/content/corpus/extra_vols.npy') else None",
    "vb = np.load(os.path.join(corpus_ext_path, 'extra_vols.npy'), mmap_mode='r') if os.path.exists(os.path.join(corpus_ext_path, 'extra_vols.npy')) else None"
)
src3 = src3.replace(
    "ma = np.load('/content/corpus/all_masks.npy')",
    "ma = np.load(os.path.join(corpus_path, 'all_masks.npy'))"
)
src3 = src3.replace(
    "mb = np.load('/content/corpus/extra_masks.npy') if os.path.exists('/content/corpus/extra_masks.npy') else np.zeros((0,) + ma.shape[1:], dtype=ma.dtype)",
    "mb = np.load(os.path.join(corpus_ext_path, 'extra_masks.npy')) if os.path.exists(os.path.join(corpus_ext_path, 'extra_masks.npy')) else np.zeros((0,) + ma.shape[1:], dtype=ma.dtype)"
)
src3 = src3.replace(
    "a_ids = np.load('/content/corpus/all_ids.npy', allow_pickle=True).astype(str)",
    "a_ids = np.load(os.path.join(corpus_path, 'all_ids.npy'), allow_pickle=True).astype(str)"
)
src3 = src3.replace(
    "b_ids = np.load('/content/corpus/extra_ids.npy', allow_pickle=True).astype(str) if os.path.exists('/content/corpus/extra_ids.npy') else np.array([])",
    "b_ids = np.load(os.path.join(corpus_ext_path, 'extra_ids.npy'), allow_pickle=True).astype(str) if os.path.exists(os.path.join(corpus_ext_path, 'extra_ids.npy')) else np.array([])"
)

nb['cells'][3]['source'] = [src3]

with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Fixed corpus paths in cell 3')