#!/bin/bash
# box.sh ARM CACHE [resume]: one calibration run (calib.py's ARM) on this box, in the
# foreground (start it in tmux). Everything under /workspace/calib: the repository (repo/),
# the runs (runs/), the logs (logs/ARM.log, logs/ARM.gpu: the GPU's state and the commit
# before and after) and the box's own compile caches (CACHE: cache-box4 or cache-box7).
# "resume" continues the run from its newest checkpoint (bnb train --resume); any further
# arguments go to bnb train (e.g. --bc-budget-steps 100000000 to continue a BC arm).
set -uo pipefail
ARM=$1; CACHE=$2; MODE=${3:-new}; shift $(( $# < 3 ? $# : 3 ))
ROOT=/workspace/calib
mkdir -p "$ROOT/$CACHE/inductor" "$ROOT/$CACHE/triton" "$ROOT/$CACHE/bnb" "$ROOT/logs" "$ROOT/runs"
export TORCHINDUCTOR_CACHE_DIR=$ROOT/$CACHE/inductor TRITON_CACHE_DIR=$ROOT/$CACHE/triton
export BNB_CACHE=$ROOT/$CACHE/bnb
export TORCHINDUCTOR_COMPILE_THREADS=4 OMP_NUM_THREADS=5 TRITON_DEFAULT_FP_FUSION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONFAULTHANDLER=1
cd "$ROOT/repo" || exit 1
gpu() { nvidia-smi --query-gpu=pstate,clocks.sm,power.draw,power.limit,temperature.gpu,memory.used --format=csv,noheader; }
{ echo "start $(date -u +%FT%TZ) $(hostname) $(gpu)"; echo "commit $(git rev-parse HEAD)"; } >> "$ROOT/logs/$ARM.gpu"
if [ "$MODE" = resume ]; then
  .venv/bin/bnb train --resume "$ARM" --runs-directory "$ROOT/runs" "$@" >> "$ROOT/logs/$ARM.log" 2>&1
else
  .venv/bin/python docs/rewrite/sections/overnight-calibration/calib.py run "$ARM" \
    --runs "$ROOT/runs" > "$ROOT/logs/$ARM.log" 2>&1
fi
echo "exit $? $(date -u +%FT%TZ) $(gpu)" >> "$ROOT/logs/$ARM.gpu"
