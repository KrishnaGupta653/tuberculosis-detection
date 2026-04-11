#!/usr/bin/env python3
import sys
import traceback

try:
    print("Attempting to import model...")
    from models import model_final_fast
    print("✅ Import successful!")
except SyntaxError as e:
    print(f"❌ SYNTAX ERROR at line {e.lineno}:")
    print(f"   Message: {e.msg}")
    if e.text:
        print(f"   Line content: {e.text.strip()}")
    print(f"   Filename: {e.filename}")
    traceback.print_exc()
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}")
    print(f"   {e}")
    traceback.print_exc()
