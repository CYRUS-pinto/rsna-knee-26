with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\colab_oof_extraction\engine12_13_all_folds_oof_extraction.ipynb', 'r') as f:
    lines = f.readlines()

# Fix lines 224 and 225 (0-indexed 223, 224) - need proper indentation and comma
lines[223] = '        "            nn.Tanh(),\\n",\n'
lines[224] = '        "            nn.Dropout(drop),\\n",\n'

print('Fixed lines 224, 225:')
print('224:', repr(lines[223]))
print('225:', repr(lines[224]))

with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\colab_oof_extraction\engine12_13_all_folds_oof_extraction.ipynb', 'w') as f:
    f.writelines(lines)

print('Fixed!')