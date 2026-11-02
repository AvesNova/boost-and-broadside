import subprocess, sys
sys.path.insert(0, "/home/vizia/avesnova/bnb-learner")
from bench.lock import bench_lock
with bench_lock():
    sys.exit(subprocess.run(sys.argv[1:]).returncode)
