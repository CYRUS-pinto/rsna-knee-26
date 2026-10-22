with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'r') as f:
    lines = f.readlines()

# Fix line 502 (index 501) - malformed JSON string
# The issue is escaped newlines in the JSON string
lines[501] = '        "        \\n        fold_preds = []\\n",\n'
lines[502] = '        "        fold_uids = []\\n",\n'

print('Fixed lines 502, 503')

with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'w') as f:
    f.writelines(lines)

print('Written!')