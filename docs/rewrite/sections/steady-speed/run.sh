#!/bin/bash
# run.sh NAME TREE CONFIG UPDATES TIMED CACHE [SET]: one steady.py run of tree TREE
# (/workspace/wt-TREE) on gpubox4, caches in /workspace/ss/cache/CACHE (MODE cold: remove
# them first, by passing CACHE=cold:DIR; EXTRA: more steady.py options), raw output logs/NAME.{json,log}, the GPU's state
# and power profile before and after in logs/NAME.gpu.
set -uo pipefail
NAME=$1; TREE=$2; CONFIG=$3; UPDATES=$4; TIMED=$5; CACHE=$6; SETS=${7:-}
ROOT=/workspace/ss
if [[ $CACHE == cold:* ]]; then CACHE=${CACHE#cold:}; rm -rf "$ROOT/cache/$CACHE"; fi
mkdir -p "$ROOT/cache/$CACHE/inductor" "$ROOT/cache/$CACHE/triton" "$ROOT/logs"
export TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/$CACHE/inductor TRITON_CACHE_DIR=$ROOT/cache/$CACHE/triton
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-5} OMP_NUM_THREADS=5
export TRITON_DEFAULT_FP_FUSION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONFAULTHANDLER=1
export PYTHONPATH=/workspace/wt-$TREE/src
gpu() { nvidia-smi --query-gpu=pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used,clocks_throttle_reasons.active --format=csv,noheader; }
{ echo "before $(date +%s) $(gpu)"; echo "commit $(git -C /workspace/wt-$TREE rev-parse --short HEAD)"; } > "$ROOT/logs/$NAME.gpu"
START=$(date +%s.%N)
/workspace/bnb/.venv/bin/python $ROOT/steady.py config="$ROOT/configs/$CONFIG" updates="$UPDATES" \
  timed="$TIMED" set="$SETS" ${EXTRA:-} process_start="$START" > "$ROOT/logs/$NAME.json" 2> "$ROOT/logs/$NAME.log"
echo "exit $? $(date +%s) $(gpu)" >> "$ROOT/logs/$NAME.gpu"
