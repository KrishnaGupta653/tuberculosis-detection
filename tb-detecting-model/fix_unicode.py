#!/usr/bin/env python3
# Comprehensive Unicode replacement

with open('models/model_final_fast.py', 'rb') as f:
    content = f.read().decode('utf-8', errors='ignore')

# Replace all problematic Unicode characters
replacements = {
    '✓': '[OK]',
    '✅': '[OK]',
    '✨': '[*]',
    '⚠️': '[!]',
    '❌': '[X]',
    '🎉': '[DONE]',
    '🚀': '[GO]',
    '→': '->',
    '✗': '[X]',
    '⚠': '[!]',
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open('models/model_final_fast.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Successfully fixed all Unicode characters')
