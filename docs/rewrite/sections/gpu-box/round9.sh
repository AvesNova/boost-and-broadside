#!/bin/bash
# Round 9 (the author, 00:05 KST): 60M of RL each for the champion and the null, warm-started from
# each BC run's 120M uniform average (chosen from tournament 6 and the scorecards). PPO's own
# settings (the spec's discounts; warm-start entropy 0), the file's build; the store on the GPU and
# 640-row micro-batches (64 views of PPO's 256-view minibatches), else 320.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
rl() {  # source BC run, RL run name
  local src=$1 name=$2 init=runs/$1/policies/step_120586240/uniform_average
  [ -f $init/checkpoint.json ] || { say "FAILED $name: no $init"; return 1; }
  if grep -q '"status": "finished"' runs/$name/run.json 2>/dev/null; then say "$name already finished"; return 0; fi
  say "start $name (RL, 60M, from $init)"
  for rows in 320; do  # 640 ran out of memory (champion, 01:00 KST)
    if .venv/bin/bnb train --name $name --init-from $init --preset mixed --updates 58 \
        --micro-batch-rows $rows --set host_store=false >> runs/$name.out 2>&1; then
      say "done $name"; docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1; return 0
    fi
    say "FAILED $name with micro-batch-rows $rows (runs/$name.out)"
    [ -d runs/$name ] && mv runs/$name runs/failed/$name-rows$rows-$(date +%s)
  done
}
rl 2026-11-09-bc-deadreckon-split-hiddenphysics 2026-11-09-rl-deadreckon-split-hiddenphysics
rl 2026-11-09-bc-absolute 2026-11-09-rl-absolute
say "round9 done"
