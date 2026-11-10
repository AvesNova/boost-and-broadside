#!/usr/bin/env bash
# One compile_baseline.py case twice on the same caches: cold (empty), then warm (what a
# repeat run in a new process pays). Usage: bench/compile_warm.sh OUT_DIR NAME CASE_ARGS...
set -u
out=$1; name=$2; shift 2
here=$(cd "$(dirname "$0")/.." && pwd)
python=${BNB_PYTHON:-/home/vizia/avesnova/boost-and-broadside-v2/.venv/bin/python}
mkdir -p "$out"
caches=$(mktemp -d "${TMPDIR:-/tmp}/bnb-warm-$name.XXXX")
for run in cold warm; do
  echo "$(date '+%F %T') start $name-$run" >> "$out/progress.log"
  TORCHINDUCTOR_CACHE_DIR=$caches/inductor TRITON_CACHE_DIR=$caches/triton BNB_CACHE=$caches/bnb \
    PYTHONPATH=$here/src "$python" "$here/bench/compile_baseline.py" "$@" \
    > "$out/$name-$run.json" 2> "$out/$name-$run.log"
  echo "$(date '+%F %T') done $name-$run exit $?" >> "$out/progress.log"
done
rm -rf "$caches"
