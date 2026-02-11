
try:
    from boost_and_broadside.env.env import Environment
    print("Import src.env.env success")
except ImportError:
    print("Import src.env.env failed")

try:
    from env.env import Environment
    print("Import env.env success")
except ImportError:
    print("Import env.env failed")

try:
    from boost_and_broadside.env2.env import TensorEnv
    print("Import src.env2.env success")
except ImportError:
    print("Import src.env2.env failed")
