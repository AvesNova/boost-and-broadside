
import sys
print(f"Path: {sys.path}")
print("Starting import...")
try:
    print("Import successful")
except Exception:
    import traceback
    traceback.print_exc()
