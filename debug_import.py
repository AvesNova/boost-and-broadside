import sys
import os

print("Current working directory:", os.getcwd())
print("sys.path:", sys.path)

try:
    import src
    print("Successfully imported src")
    from src.env.ship import Ship
    print("Successfully imported Ship")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
