with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'r') as f:
    content = f.read()

# Fix all malformed JSON strings with pattern: \\n        "
# The issue is that the JSON string contains literal \n followed by spaces and "
# which should be properly escaped as \n        " in the JSON string

import re

# Pattern: \\n        " (backslash-n, 8 spaces, quote) in a JSON string
# This should be \\n        " (properly escaped for JSON)
# The regex finds: \\\\n        " and replaces with \\\\n        "
# But in the raw file, it's stored as: \\n        "  (which is wrong JSON)
# The correct JSON should have: \\n        " (properly escaped)

# Let's do a more careful fix - find all occurrences of the pattern in JSON strings
# The issue is in the source arrays of code cells where strings contain \\n        "
# In valid JSON, this should be: \\n        " (double-escaped for JSON)

# Read line by line and fix
with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'r') as f:
    lines = f.readlines()

fixed_lines = []
for line in lines:
    # Fix pattern: "        \\n        "  -> "        \\n        "
    # But careful: in the JSON source, this appears as escaped
    # The raw line contains: "        \\n        "        "
    # Which should be: "        \\n        "
    
    # Pattern: \\n        "        " (in the raw line)
    # This is a string that in JSON should be: \\n        "
    # But the JSON encoder double-escapes
    
    # The issue is: the source array contains strings like:
    # "        \\n        "        fold_preds = []\\n",
    # But valid JSON should have:
    # "        \\n        fold_preds = []\\n",
    
    # So we need to remove the extra "        " after \\n
    
    new_line = line
    # Pattern: \\n        "        "  -> \\n        "
    # In the raw string: \\n        "        "
    # This appears as: \\n        "        " in the Python string
    # Which is: backslash, n, 8 spaces, quote, 8 spaces, quote
    # We want: backslash, n, 8 spaces, quote
    
    import re
    # Match: \\n        "        " (literal backslash-n, 8 spaces, quote, 8 spaces, quote)
    # Replace with: \\n        "
    new_line = re.sub(r'\\\\n        "        "', r'\\\\n        "', new_line)
    
    # Also handle: \\n        "        \\n" -> \\n        \\n"
    new_line = re.sub(r'\\\\n        "\\\\n"', r'\\\\n        \\\\n"', new_line)
    
    # Also handle: \\n        "        " at end of string
    new_line = re.sub(r'\\\\n        "        "$', r'\\\\n        "', new_line)
    
    fixed_lines.append(new_line)

# Write back
with open(r'C:\Users\Cyrus\Downloads\rsna knee fip\submissions\v50_oof_optimized_quad_arm\rsna-knee-v50-oof-optimized-quad-arm.ipynb', 'w') as f:
    f.writelines(fixed_lines)

print('Fixed all lines!')