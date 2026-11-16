#!/bin/bash
# The 8 balance conditions timed on the RTX 5090 (2026-11-16): one steady batch each at 4096
# environments and 32 variants (`bench.balance_sweep timing`), then deathmatch 5v5 and
# Frontline 10v10 at 16384 environments (does a larger batch fill the 5090?). Compiles run one
# at a time, outside the GPU lock; the lock is held for the timed batch only.
source /workspace/env.sh
here=/workspace/wt/balance-eval
out=/workspace/out/balance-eval
export PYTHONPATH=$here/src:$here TORCH_LOGS=recompiles
cd $here
nvidia-smi --query-gpu=name,driver_version,power.limit,power.max_limit,clocks.max.sm,clocks.max.mem,persistence_mode --format=csv > $out/nvidia-smi-state.txt
run() {
  condition=$1; environments=$2
  echo "$(date +%H:%M:%S) timing $condition B=$environments"
  /workspace/venv/bin/python -m bench.balance_sweep timing --condition $condition \
    --environments $environments --variants 32 --lock $GPU_LOCK --out $out/timing-5090.jsonl \
    > $out/timing-$condition-$environments.log 2>&1
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,pstate,temperature.gpu --format=csv,noheader >> $out/nvidia-smi-after.txt
  tail -1 $out/timing-5090.jsonl | cut -c1-160
}
for condition in deathmatch-5v5 deathmatch-10v10 deathmatch-2v2 deathmatch-1v1 \
    elimination-5v5 elimination-10v10 frontline-5v5 frontline-10v10; do
  run $condition 4096
done
run deathmatch-5v5 16384
run frontline-10v10 16384
echo "$(date +%H:%M:%S) done"
