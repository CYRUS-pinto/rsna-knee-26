import json

with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'r') as f:
    nb = json.load(f)

# Fix cell 2 - update all dataset paths to use correct Kaggle mount paths
cell2 = nb['cells'][2]
src = ''.join(cell2['source'])

# Update all dataset paths to use correct Kaggle mount slugs
src = src.replace(
    'weight_paths = {\n',
    'weight_paths = {\n'
)
# The weight paths are already correct (they use cyruspinto/... which should work)

# Update corpus paths - use correct Kaggle dataset slugs
src = src.replace(
    'corpus_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus"',
    'corpus_path = "/kaggle/input/knee-raptor-corpus"'
)
src = src.replace(
    'corpus_ext_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext"',
    'corpus_ext_path = "/kaggle/input/knee-raptor-corpus-ext"'
)
src = src.replace(
    'labels_path = "/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916"',
    'labels_path = "/kaggle/input/rsna-knee-super-consensus-labels-0916"'
)
src = src.replace(
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"',
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"'
)

# Also update the weight_paths to use correct slugs
src = src.replace(
    '    "e12_main": "/kaggle/input/rsna-knee-engine12-convnext-large-weights",',
    '    "e12_main": "/kaggle/input/rsna-knee-engine12-convnext-large-weights",'
)
src = src.replace(
    '    "e13_main": "/kaggle/input/rsna-knee-engine13-swin-base-weights",',
    '    "e13_main": "/kaggle/input/rsna-knee-engine13-swin-base-weights",'
)
src = src.replace(
    '    "e13_fold4": "/kaggle/input/rsna-knee-engine13-fold4-weights",',
    '    "e13_fold4": "/kaggle/input/rsna-knee-engine13-fold4-weights",'
)
src = src.replace(
    '    "e10_fold0": "/kaggle/input/rsna-knee-engine10-fold0-weights",',
    '    "e10_fold0": "/kaggle/input/rsna-knee-engine10-fold0-weights",'
)
src = src.replace(
    '    "e10_fold1": "/kaggle/input/rsna-knee-engine10-fold1-weights",',
    '    "e10_fold1": "/kaggle/input/rsna-knee-engine10-fold1-weights",'
)
src = src.replace(
    '    "e10_fold2": "/kaggle/input/rsna-knee-engine10-fold2-weights",',
    '    "e10_fold2": "/kaggle/input/rsna-knee-engine10-fold2-weights",'
)
src = src.replace(
    '    "e10_fold3": "/kaggle/input/rsna-knee-engine10-fold3-weights",',
    '    "e10_fold3": "/kaggle/input/rsna-knee-engine10-fold3-weights",'
)
src = src.replace(
    '    "e10_fold4": "/kaggle/input/rsna-knee-engine10-fold4-weights",',
    '    "e10_fold4": "/kaggle/input/rsna-knee-engine10-fold4-weights",'
)

# Update corpus paths
src = src.replace(
    'corpus_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus"',
    'corpus_path = "/kaggle/input/knee-raptor-corpus"'
)
src = src.replace(
    'corpus_ext_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext"',
    'corpus_ext_path = "/kaggle/input/knee-raptor-corpus-ext"'
)
src = src.replace(
    'labels_path = "/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916"',
    'labels_path = "/kaggle/input/rsna-knee-super-consensus-labels-0916"'
)
src = src.replace(
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"',
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"'
)

# Update the weight paths to use correct cyruspinto dataset slugs
# These should already be correct as they use cyruspinto/... format
# But the actual Kaggle mount path for cyruspinto datasets is /kaggle/input/cyruspinto/<dataset-slug>/
# The dataset slug for cyruspinto/rsna-knee-engine12-convnext-large-weights is rsna-knee-engine12-convnext-large-weights
# So the path should be /kaggle/input/cyruspinto/rsna-knee-engine12-convnext-large-weights/
# But in the code, we use /kaggle/input/rsna-knee-engine12-convnext-large-weights which is wrong
# The correct path is /kaggle/input/cyruspinto/rsna-knee-engine12-convnext-large-weights/

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

# Also update the corpus paths
src = src.replace(
    'corpus_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus"',
    'corpus_path = "/kaggle/input/knee-raptor-corpus"'
)
src = src.replace(
    'corpus_ext_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext"',
    'corpus_ext_path = "/kaggle/input/knee-raptor-corpus-ext"'
)
src = src.replace(
    'labels_path = "/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916"',
    'labels_path = "/kaggle/input/rsna-knee-super-consensus-labels-0916"'
)
src = src.replace(
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"',
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"'
)

# Also update the weight_paths to use cyruspinto prefix
src = src.replace(
    '    "e12_main": "/kaggle/input/rsna-knee-engine12-convnext-large-weights",',
    '    "e12_main": "/kaggle/input/cyruspinto/rsna-knee-engine12-convnext-large-weights",'
)
src = src.replace(
    '    "e13_main": "/kaggle/input/rsna-knee-engine13-swin-base-weights",',
    '    "e13_main": "/kaggle/input/cyruspinto/rsna-knee-engine13-swin-base-weights",'
)
src = src.replace(
    '    "e13_fold4": "/kaggle/input/rsna-knee-engine13-fold4-weights",',
    '    "e13_fold4": "/kaggle/input/cyruspinto/rsna-knee-engine13-fold4-weights",'
)
src = src.replace(
    '    "e10_fold0": "/kaggle/input/rsna-knee-engine10-fold0-weights",',
    '    "e10_fold0": "/kaggle/input/cyruspinto/rsna-knee-engine10-fold0-weights",'
)
src = src.replace(
    '    "e10_fold1": "/kaggle/input/rsna-knee-engine10-fold1-weights",',
    '    "e10_fold1": "/kaggle/input/cyruspinto/rsna-knee-engine10-fold1-weights",'
)
src = src.replace(
    '    "e10_fold2": "/kaggle/input/rsna-knee-engine10-fold2-weights",',
    '    "e10_fold2": "/kaggle/input/cyruspinto/rsna-knee-engine10-fold2-weights",'
)
src = src.replace(
    '    "e10_fold3": "/kaggle/input/rsna-knee-engine10-fold3-weights",',
    '    "e10_fold3": "/kaggle/input/cyruspinto/rsna-knee-engine10-fold3-weights",'
)
src = src.replace(
    '    "e10_fold4": "/kaggle/input/rsna-knee-engine10-fold4-weights",',
    '    "e10_fold4": "/kaggle/input/cyruspinto/rsna-knee-engine10-fold4-weights",'
)

# Update the corpus paths in the cell
src = src.replace(
    'corpus_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus"',
    'corpus_path = "/kaggle/input/knee-raptor-corpus"'
)
src = src.replace(
    'corpus_ext_path = "/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext"',
    'corpus_ext_path = "/kaggle/input/knee-raptor-corpus-ext"'
)
src = src.replace(
    'labels_path = "/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916"',
    'labels_path = "/kaggle/input/rsna-knee-super-consensus-labels-0916"'
)
src = src.replace(
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"',
    'comp_path = "/kaggle/input/rsna-knee-abnormality-detection"'
)

cell2 = nb['cells'][2]
src = ''.join(cell2['source'])

# Apply all the replacements
src = src.replace(
    'weight_paths = {',
    'weight_paths = {'
)

# Do the replacements
replacements = [
    ('/kaggle/input/rsna-knee-engine12-convnext-large-weights', '/kaggle/input/cyruspinto/rsna-knee-engine12-convnext-large-weights'),
    ('/kaggle/input/rsna-knee-engine13-swin-base-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-swin-base-weights'),
    ('/kaggle/input/rsna-knee-engine13-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine13-fold4-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold0-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold0-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold1-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold1-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold2-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold2-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold3-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold3-weights'),
    ('/kaggle/input/rsna-knee-engine10-fold4-weights', '/kaggle/input/cyruspinto/rsna-knee-engine10-fold4-weights'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus', '/kaggle/input/knee-raptor-corpus'),
    ('/kaggle/input/dreaddevelopment/knee-raptor-corpus-ext', '/kaggle/input/knee-raptor-corpus-ext'),
    ('/kaggle/input/cyruspinto/rsna-knee-super-consensus-labels-0916', '/kaggle/input/rsna-knee-super-consensus-labels-0916'),
]

for old, new in replacements:
    src = src.replace(old, new)

cell2 = nb['cells'][2]
cell2['source'] = [src]

with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Fixed all dataset paths in cell 2')