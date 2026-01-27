
import sys
import os
sys.path.append(os.getcwd())
try:
    from src.env.env import Environment
    print("Import src.env.env success")
except ImportError:
    print("Import src.env.env failed")

try:
    from env.env import Environment
    print("Import env.env success")
except ImportError:
    print("Import env.env failed")

try:
    from src.env2.env import TensorEnv
    print("Import src.env2.env success")
except ImportError:
    print("Import src.env2.env failed")
