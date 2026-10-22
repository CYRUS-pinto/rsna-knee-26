# Fix the malformed JSON in the notebook
with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\colab_oof_extraction\engine12_13_all_folds_oof_extraction.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and fix the problematic lines around nn.Tanh and nn.Dropout
for i, line in enumerate(lines):
    if 'nn.Tanh()' in line and i > 220 and i < 230:
        print(f'Line {i+1}: {repr(line)}')
    if 'nn.Dropout(drop)' in line and i > 220 and i < 230:
        print(f'Line {i+1}: {repr(line)}')

# Fix line 224 (nn.Tanh) and 225 (nn.Dropout) - 0-indexed
# The issue is these lines have malformed JSON strings
# Line 224 (index 223): should be "            nn.Tanh(),\n"
# Line 225 (index 224): should be "            nn.Dropout(drop),\n"

# Check current state
print("Before fix:")
print(f"Line 224: {repr(lines[223])}")
print(f"Line 225: {repr(lines[224])}")

# Fix them
lines[223] = '            "            nn.Tanh(),\\n"\n'
lines[224] = '            "            nn.Dropout(drop),\\n"\n'

print("After fix:")
print(f"Line 224: {repr(lines[223])}")
print(f"Line 225: {repr(lines[224])}")

# Write back
with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\colab_oof_extraction\engine12_13_all_folds_oof_extraction.ipynb', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Fixed!')