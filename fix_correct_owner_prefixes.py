import json

with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'r') as f:
    nb = json.load(f)

# Fix cell 2 - revert corpus paths to use owner prefix
cell2 = nb['cells'][2]
src = ''.join(cell2['source'])

# Fix corpus paths - use owner prefix
src = src.replace(
    'corpus_path = "/kaggle/input/knee-raptor-corpus"',
    'corpus_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus"'
)
src = src.replace(
    'corpus_ext_path = "/kaggle/input/knee-raptor-corpus-ext"',
    'corpus_ext_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext"'
)

# Also fix the labels path - it should be cyruspinto prefix
src = src.replace(
    'labels_path = "/kaggle/input/rsna-knee-super-consensus-labels-0916"',
    'labels_path = "/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916"'
)

# The competition path should be correct
src = src.replace(
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"',
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"'
)

# Also fix the weight paths - they should use cyruspinto prefix
# But the weight datasets are owned by cyruspinto, so they should be /kaggle/input/cyruspinto/...
# The dataset slugs for weights are rsna-knee-engine12-convnext-large-weights etc.
# So the path should be /kaggle/input/cyruspinto/rsna-knee-engine12-convnext-large-weights/

cell2 = nb['cells'][2]
src = ''.join(cell2['source'])

# Fix all dataset paths to use correct owner prefixes
replacements = [
    ('/kaggle/input/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
]

for old, new in [
    ('/kaggle/input/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
]:
    src = src.replace(old, new)

# Also fix the weight paths to use cyruspinto prefix
src = src.replace(
    '/kaggle/input/rsna-knee-engine12-convnext-large-weights',
    '/kaggle/input/cyruspinto/rsna-knee-engine12-convnext-large-weights'
)
src = src.replace(
    '/kaggle/input/rsna-knee-engine13-swin-base-weights',
    '/kaggle/input/cyruspinto/rsna-knee-engine13-swin-base-weights'
)
src = src.replace(
    '/kaggle/input/rsna-knee-engine13-fold4-weights',
    '/kaggle/input/cyruspinto/rsna-knee-engine13-fold4-weights'
)
src = src.replace(
    '/kaggle/input/rsna-knee-engine10-fold0-weights',
    '/kaggle/input/cyruspinto/rsna-knee-engine10-fold0-weights'
)
src = src.replace(
    '/kaggle/input/rsna-knee-engine10-fold1-weights',
    '/kaggle/input/cyruspinto/rsna-knee-engine10-fold1-weights'
)
src = src.replace(
    '/kaggle/input/rsna-knee-engine10-fold1-weights',
    '/kaggle/input/cyruspinto/rsna-knee-engine10-fold1-weights'
)
src = src.replace(
    '/kaggle/input/rsna-knee-engine10-fold2-weights',
    '/kaggle/input/cyruspinto/rsna-knee-engine10-fold2-weights'
)
src = src.replace(
    '/kaggle/input/rsna-knee-engine10-fold3-weights',
    '/kaggle/input/cyruspinto/rsna-knee-engine10-fold3-weights'
)
src = src.replace(
    '/kaggle/input/rsna-knee-engine10-fold4-weights',
    '/kaggle/input/cyruspinto/rsna-knee-engine10-fold4-weights'
)
src = src.replace(
    '/kaggle/input/rsna-knee-engine10-fold4-weights',
    '/kaggle/input/cyruspinto/rsna-knee-engine10-fold4-weights'
)
src = src.replace(
    '/kaggle/input/dreaddevelopment/knee-raptor-corpus',
    '/kaggle/input/dreaddevelopment/knee-raptor-corpus'
)
src = src.replace(
    '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext',
    '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'
)
src = src.replace(
    '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916',
    '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'
)

cell2 = nb['cells'][2]
src = ''.join(cell2['source'])

# Apply replacements
replacements = [
    ('/kaggle/input/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/rsna-knee-engine12-convnext-large-weights', '/kaggle/input/cyruspinto/rsna-knee-engine12-convnext-large-weights'),
    ('/kaggle/input/rsna-knee-engine13-swin-base-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-swin-base-weights'),
    ('/kaggle/input/rsna-knee-engine13-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-fold4-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold0-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold0-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold1-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold1-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold2-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold2-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold3-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold3-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold4-weights'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
]

for old, new in [
    ('/kaggle/input/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/rsna-knee-engine12-convnext-large-weights', '/kaggle/input/cyruspinto/rsna-knee-engine12-convnext-large-weights'),
    ('/kaggle/input/rsna-knee-engine13-swin-base-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-swin-base-weights'),
    ('/kaggle/input/rsna-knee-engine13-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-fold4-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold0-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold0-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold1-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold1-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold2-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold2-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold3-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold3-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold4-weights'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
]:
    src = src.replace(old, new)

# Also fix the corpus paths
src = src.replace(
    'corpus_path = "/kaggle/input/knee-raptor-corpus"',
    'corpus_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus"'
)
src = src.replace(
    'corpus_ext_path = "/kaggle/input/knee-raptor-corpus-ext"',
    'corpus_ext_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext"'
)
src = src.replace(
    'labels_path = "/kaggle/input/rsna-knee-super-consensus-labels-0916"',
    'labels_path = "/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916"'
)

cell2 = nb['cells'][2]
src = ''.join(cell2['source'])

# Apply all replacements
replacements = [
    ('/kaggle/input/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/rsna-knee-engine12-convnext-large-weights', '/kaggle/input/cyruspinto/rsna-knee-engine12-convnext-large-weights'),
    ('/kaggle/input/rsna-knee-engine13-swin-base-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-swin-base-weights'),
    ('/kaggle/input/rsna-knee-engine13-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-fold4-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold0-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold0-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold1-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold1-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold2-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold2-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold3-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold3-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold4-weights'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
]

for old, new in [
    ('/kaggle/input/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/rsna-knee-engine12-convnext-large-weights', '/kaggle/input/cyruspinto/rsna-knee-engine12-convnext-large-weights'),
    ('/kaggle/input/rsna-knee-engine13-swin-base-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-swin-base-weights'),
    ('/kaggle/input/rsna-knee-engine13-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-fold4-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold0-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold0-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold1-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold1-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold2-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold2-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold3-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold3-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold4-weights'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
]:
    src = src.replace(old, new)

# Also fix the corpus paths
src = src.replace(
    'corpus_path = "/kaggle/input/knee-raptor-corpus"',
    'corpus_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus"'
)
src = src.replace(
    'corpus_ext_path = "/kaggle/input/knee-raptor-corpus-ext"',
    'corpus_ext_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext"'
)
src = src.replace(
    'labels_path = "/kaggle/input/rsna-knee-super-consensus-labels-0916"',
    'labels_path = "/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916"'
)
src = src.replace(
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"',
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"'
)

cell2 = nb['cells'][2]
src = ''.join(cell2['source'])

# Apply all replacements
replacements = [
    ('/kaggle/input/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/rsna-knee-engine12-convnext-large-weights', '/kaggle/input/cyruspinto/rsna-knee-engine12-convnext-large-weights'),
    ('/kaggle/input/rsna-knee-engine13-swin-base-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-swin-base-weights'),
    ('/kaggle/input/rsna-knee-engine13-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-fold4-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold0-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold0-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold1-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold1-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold2-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold2-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold3-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold3-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold4-weights'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
]

for old, new in [
    ('/kaggle/input/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/rsna-knee-engine12-convnext-large-weights', '/kaggle/input/cyruspinto/rsna-knee-engine12-convnext-large-weights'),
    ('/kaggle/input/rsna-knee-engine13-swin-base-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-swin-base-weights'),
    ('/kaggle/input/rsna-knee-engine13-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-fold4-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold0-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold0-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold1-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold1-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold2-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold2-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold3-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold3-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold4-weights'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
]:
    src = src.replace(old, new)

# Also fix the corpus paths
src = src.replace(
    'corpus_path = "/kaggle/input/knee-raptor-corpus"',
    'corpus_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus"'
)
src = src.replace(
    'corpus_ext_path = "/kaggle/input/knee-raptor-corpus-ext"',
    'corpus_ext_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext"'
)
src = src.replace(
    'labels_path = "/kaggle/input/rsna-knee-super-consensus-labels-0916"',
    'labels_path = "/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916"'
)
src = src.replace(
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"',
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"'
)

cell2 = nb['cells'][2]
src = ''.join(cell2['source'])

# Apply all replacements
replacements = [
    ('/kaggle/input/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/rsna-knee-engine12-convnext-large-weights', '/kaggle/input/cyruspinto/rsna-knee-engine12-convnext-large-weights'),
    ('/kaggle/input/rsna-knee-engine13-swin-base-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-swin-base-weights'),
    ('/kaggle/input/rsna-knee-engine13-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-fold4-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold0-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold0-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold1-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold1-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold2-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold2-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold3-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold3-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold4-weights'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
]

for old, new in [
    ('/kaggle/input/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/rsna-knee-engine12-convnext-large-weights', '/kaggle/input/cyruspinto/rsna-knee-engine12-convnext-large-weights'),
    ('/kaggle/input/rsna-knee-engine13-swin-base-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-swin-base-weights'),
    ('/kaggle/input/rsna-knee-engine13-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-fold4-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold0-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold0-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold1-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold1-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold2-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold2-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold3-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold3-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold4-weights'),
    ('/kaggle/input/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus', '/kaggle/input/dreaddevelopment/knee-raptor-corpus'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext', '/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext'),
    ('/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916', '/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916'),
]:
    src = src.replace(old, new)

# Also fix the corpus paths
src = src.replace(
    'corpus_path = "/kaggle/input/knee-raptor-corpus"',
    'corpus_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus"'
)
src = src.replace(
    'corpus_ext_path = "/kaggle/input/knee-raptor-corpus-ext"',
    'corpus_ext_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext"'
)
src = src.replace(
    'labels_path = "/kaggle/input/rsna-knee-super-consensus-labels-0916"',
    'labels_path = "/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916"'
)
src = src.replace(
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"',
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"'
)

cell2 = nb['cells'][2]
cell2['source'] = [src]

with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Fixed all dataset paths with correct owner prefixes')