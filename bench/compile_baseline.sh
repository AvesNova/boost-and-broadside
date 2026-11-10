#!/usr/bin/env bash
# Run compile_baseline.py's cases one after another, each in its own process on fresh
# (empty) inductor and Triton caches. Usage: bench/compile_baseline.sh OUT_DIR CASE_SPEC...
# where CASE_SPEC is "name:case key=value ..." (quoted). Results: OUT_DIR/<name>.json, logs
# OUT_DIR/<name>.log, progress OUT_DIR/progress.log.
set -u
out=$1; shift
here=$(cd "$(dirname "$0")/.." && pwd)
python=${BNB_PYTHON:-/home/vizia/avesnova/boost-and-broadside-v2/.venv/bin/python}
mkdir -p "$out"
for spec in "$@"; do
  name=${spec%%:*}; arguments=${spec#*:}
  caches=$(mktemp -d "${TMPDIR:-/tmp}/bnb-cold-$name.XXXX")
  echo "$(date '+%F %T') start $name ($arguments) caches $caches" >> "$out/progress.log"
  TORCHINDUCTOR_CACHE_DIR=$caches/inductor TRITON_CACHE_DIR=$caches/triton BNB_CACHE=$caches/bnb \
    PYTHONPATH=$here/src "$python" "$here/bench/compile_baseline.py" $arguments \
    > "$out/$name.json" 2> "$out/$name.log"
  status=$?
  echo "$(date '+%F %T') done $name exit $status" >> "$out/progress.log"
  rm -rf "$caches"
done
