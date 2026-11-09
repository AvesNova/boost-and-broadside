#!/bin/bash
# Round 8 (the author, 19:40 KST): the champion (deadreckon-split-hiddenphysics) and the null
# (absolute) to 120M BC (resumed from 60.8M), then 60M of RL each, warm-started from each run's
# 120M uniform average (the policy kind the tournaments rated), PPO's own settings (the spec's
# discounts; warm-start entropy 0), the same model build as the file's.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
B=2026-11-09-bc
BC120=125829120   # 120 updates of 1,048,576 steps: the first snapshot at or past 120M
for name in $B-deadreckon-split-hiddenphysics $B-absolute; do
  if ls -d runs/$name/policies/step_1[2-9]* > /dev/null 2>&1; then say "$name already past 120M"; continue; fi
  say "start resume $name to 120M BC"
  if .venv/bin/bnb train --resume $name --bc-budget-steps 120000000 >> runs/$name.out 2>&1; then
    say "done resume $name to 120M"
  else
    say "FAILED resume $name to 120M (runs/$name.out)"
  fi
  docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
done

rl() {  # source BC run, RL run name
  local src=$1 name=$2 final
  final=$(ls -d runs/$src/policies/step_*/uniform_average | sort -V | tail -1)
  say "start $name (RL, 60M, from $final)"
  for rows in 320 160; do
    if .venv/bin/bnb train --name $name --init-from $final --preset mixed --updates 58 \
        --micro-batch-rows $rows --set host_store=false >> runs/$name.out 2>&1; then
      say "done $name"; docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1; return 0
    fi
    say "FAILED $name with micro-batch-rows $rows (runs/$name.out)"
    [ -d runs/$name ] && mv runs/$name runs/failed/$name-rows$rows-$(date +%s)
  done
}
rl $B-deadreckon-split-hiddenphysics 2026-11-09-rl-deadreckon-split-hiddenphysics
rl $B-absolute 2026-11-09-rl-absolute
say "round8 done"
