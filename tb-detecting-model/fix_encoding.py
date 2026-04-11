#!/usr/bin/env python3
# Fix all Unicode encoding issues

with open(r'models/model_final_fast.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all problematic Unicode characters with ASCII equivalents
replacements = {
    '✓': '[OK]',
    '✅': '[OK]',
    '✨': '[*]',
    '⚠️': '[!]',
    '❌': '[X]',
    '🎉': '[*]',
    '🚀': '[RUN]',
    '→': '->',
    '✗': '[X]',
}

for unicode_char, ascii_equiv in replacements.items():
    content = content.replace(unicode_char, ascii_equiv)

with open(r'models/model_final_fast.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[OK] All Unicode characters replaced with ASCII equivalents")
