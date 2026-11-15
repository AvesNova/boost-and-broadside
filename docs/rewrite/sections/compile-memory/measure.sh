#!/bin/bash
# One cold compile measurement on the rented box: the full preset's training command for
# UPDATES updates, on fresh caches, with tree_memory.py sampling the process tree.
# Usage: measure.sh NAME TREE THREADS ENVIRONMENTS UPDATES [--set ...]
# THREADS: TORCHINDUCTOR_COMPILE_THREADS ("default": unset). Run it under flock /tmp/gpu.lock.
name=$1; tree=$2; threads=$3; environments=$4; updates=$5; shift 5
out=/workspace/cm-results/$name
cache=/workspace/cm-cache-$name
rm -rf "$out" "$cache"; mkdir -p "$out" "$cache"
cd "$tree"
export PYTHONPATH=$tree/src TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton BNB_CACHE=$cache/bnb
[ "$threads" != default ] && export TORCHINDUCTOR_COMPILE_THREADS=$threads
{
  echo "name=$name tree=$tree commit=$(git rev-parse --short HEAD) threads=$threads environments=$environments updates=$updates extra=$*"
  echo "start $(date +%T) $(nvidia-smi --query-gpu=pstate,power.limit,clocks.sm --format=csv,noheader)"
  .venv/bin/python -c "import bnb; print('bnb', bnb.__file__)"
} > "$out/meta.txt"
.venv/bin/python -c "import bnb.cli; bnb.cli.main()" train --preset real_scale_all_modes \
  --environments "$environments" --updates "$updates" --micro-batch-rows 96 --name "$name" \
  --runs-directory "$out/runs" --set lanes.environments=32 --set host_store=true \
  --set arena_scale=null "$@" > "$out/train.log" 2>&1 &
pid=$!
/workspace/compile-mem/.venv/bin/python /workspace/cm-tools/tree_memory.py \
  $pid "$out/memory.csv" "$cache" 2 > "$out/peaks.txt" 2>&1 &
wait $pid
status=$?
wait
{
  echo "exit $status $(date +%T)"
  echo "graphs $(find "$cache" -name 'output_code*' -o -path '*inductor*' -name '*.py' -newer "$out/meta.txt" | wc -l)"
  echo "cubins $(find "$cache" -name '*.cubin' | wc -l)"
  echo "cache $(du -sh "$cache" | cut -f1)"
} >> "$out/meta.txt"
rm -rf "$cache"
