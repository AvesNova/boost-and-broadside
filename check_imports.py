
import sys
import os

print(f"CWD: {os.getcwd()}")
print("sys.path:")
for p in sys.path:
    print(p)

try:
    import boost_and_broadside.env2
    print("SUCCESS: import env2")
except ImportError as e:
    print(f"FAILURE: import env2 -> {e}")

try:
    import boost_and_broadside.env2
    print("SUCCESS: import env2")
except ImportError as e:
    print(f"FAILURE: import env2 -> {e}")

try:
    import boost_and_broadside.core.constants
    print("SUCCESS: import core.constants")
except ImportError as e:
    print(f"FAILURE: import core.constants -> {e}")

try:
    from env.ship import Ship
    print("SUCCESS: from env.ship import Ship")
except ImportError as e:
    print(f"FAILURE: from env.ship import Ship -> {e}")
