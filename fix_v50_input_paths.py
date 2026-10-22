import json

# Fix the V50 main notebook to use /kaggle/input/ paths instead of kaggle CLI downloads
with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'r') as f:
    nb = json.load(f)

# Replace cell 2 (download cell) with version that uses /kaggle/input/ paths
cell2 = nb['cells'][2]
new_source = [
    '# 2. Load Assets from Kaggle Input Paths (No CLI Download Needed)\n',
    'import os, glob\n',
    'print("Loading assets from /kaggle/input/...")\n',
    '\n',
    '# Weight datasets are mounted at /kaggle/input/<dataset-slug>/\n',
    'weight_paths = {\n',
    '    "e12_main": "/kaggle/input/rsna-knee-engine12-convnext-large-weights",\n',
    '    "e13_main": "/kaggle/input/rsna-knee-engine13-swin-base-weights",\n',
    '    "e13_fold4": "/kaggle/input/rsna-knee-engine13-fold4-weights",\n',
    '    "e10_fold0": "/kaggle/input/rsna-knee-engine10-fold0-weights",\n',
    '    "e10_fold1": "/kaggle/input/rsna-knee-engine10-fold0-weights",\n',
    '    "e10_fold1": "/kaggle/input/rsna-knee-engine10-fold1-weights",\n',
    '    "e10_fold2": "/kaggle/input/rsna-knee-engine10-fold2-weights",\n',
    '    "e10_fold3": "/kaggle/input/rsna-knee-engine10-fold3-weights",\n',
    '    "e10_fold4": "/kaggle/input/rsna-knee-engine10-fold4-weights",\n',
    '}\n',
    'for name, path in weight_paths.items():\n',
    '    pts = glob.glob(os.path.join(path, "**/*.pt"), recursive=True) + glob.glob(os.path.join(path, "**/*.pth"), recursive=True)\n',
    '    if pts:\n',
    '        print(f"[+] {name}: {[os.path.basename(p) for p in pts]}")\n',
    '    else:\n',
    '    print(f"[!] {name}: No .pt files found at {path}")\n',
    '\n',
    '# Corpus & labels\n',
    'corpus_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus"\n',
    'corpus_ext_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext"\n',
    'labels_path = "/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916"\n',
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"\n',
    '\n',
    'print("All assets loaded from /kaggle/input/")\n',
]
cell2['source'] = new_source

with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Fixed V50 main notebook to use /kaggle/input/ paths')