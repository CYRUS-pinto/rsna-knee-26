import json
import os

variants = [
    r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_variant_a_low_anchor\rsna-knee-v50-variant-a.ipynb',
    r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_variant_b_high_anchor\rsna-knee-v50-variant-b.ipynb',
    r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_variant_c_no_raptor\rsna-knee-v50-variant-c.ipynb',
    r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_variant_d_alt_weights\rsna-knee-v50-variant-d.ipynb',
]

for path in variants:
    with open(path, 'r') as f:
        nb = json.load(f)

    if len(nb['cells']) > 3:
        cell3 = nb['cells'][3]
        src3 = ''.join(cell3['source'])

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

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'Fixed {path}')

print('All variants fixed!')