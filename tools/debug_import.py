
import sys
import os
sys.path.append(os.getcwd())
print(f"Path: {sys.path}")
print("Starting import...")
try:
    from src.env2.env import TensorEnv
    print("Import successful")
except Exception as e:
    import traceback
    traceback.print_exc()
