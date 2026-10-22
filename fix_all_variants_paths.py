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

    if len(nb['cells']) > 2:
        cell2 = nb['cells'][2]
        src = ''.join(cell2['source'])

        # Fix all dataset paths
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

        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                src = ''.join(cell['source'])
                for old, new in [
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
                ]:
                    src = src.replace(old, new)

                # Also fix the corpus paths in cell 3
                if 'corpus_path' in src:
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

                cell['source'] = [src]

        with open(path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f'Fixed {path}')

print('All variants fixed!')