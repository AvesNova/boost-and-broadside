#!/bin/bash
# queue.sh [STEP...]: the steady-speed measurements not run before the 01:00 stop of
# 2026-11-12, one GPU process at a time on gpubox4 (two learn phases do not fit 16 GB at
# 1024 environments). Run from /workspace/ss after `git -C /workspace/wt-final checkout
# --detach origin/perf/steady-speed` and copying configs.sh, run.sh, steady.py,
# summarise.py here. Each step appends nothing itself: summarise afterwards with
#   python summarise.py gpubox4 NAME logs/NAME.json logs/NAME.gpu skip=2 >> records.jsonl
# Steps (default all, in this order):
#   configs   final-tree configs at 1024 environments (RL 320 rows; BC 2560 rows, no lanes)
#   rl        final RL (new lanes, the pool's scripted entry, heads' sums sample), 12
#             updates: confirms the cadence update drops to ~98 s; its rollout against
#             AFTER RL's 16.25 s gives the new lanes' and the scripted pass's cost together
#   scripted  the always-on scripted pass alone: lanes 0 in the old tree (cbf0584e) and the
#             final tree, 6 updates each; the rollout difference is the pool's scripted pass
#   lanes     final RL at 512, 256 lane environments (with rl: 1024; with scripted: 0)
#   bc        final BC (no lanes, 2560 rows, drift fix), cold, 12 updates
#   memory    RL host_store true vs false at 320 rows; RL 640 vs 320 rows
#   laptop    the 5090's config (2048 environments, BC 256 / RL 32 minibatches) under an
#             8 GB cap (memory_fraction 0.49): 2 updates each, rows 1280 / 640 / 320 tried
set -uo pipefail
cd /workspace/ss
STEPS=${*:-configs rl scripted lanes bc memory laptop}
for step in $STEPS; do case $step in
configs)
  ./configs.sh final configs/final 1024 learner.micro_batch_rows=2560 lanes.environments=0
  mv configs/final/bc.json configs/final/bc-nolanes.json
  ./configs.sh final configs/final 1024 learner.micro_batch_rows=320
  for rows in 1280 640 320; do
    ./configs.sh final configs/final2048-$rows 2048 learner.micro_batch_rows=$rows lanes.environments=0
  done ;;
rl)
  mkdir -p cache/final-ppo && cp -rn cache/after-ppo/. cache/final-ppo/  # the learner's graphs
  ./run.sh final-ppo final final/ppo.json 12 0 final-ppo ;;
scripted)
  ./run.sh after-ppo-nolanes after after/ppo.json 6 0 after-ppo-nolanes 'lanes.environments=0'
  ./run.sh final-ppo-nolanes final final/ppo.json 6 0 final-ppo-nolanes 'lanes.environments=0' ;;
lanes)
  for lanes in 512 256; do
    ./run.sh final-ppo-lanes$lanes final final/ppo.json 12 0 final-ppo-lanes$lanes "lanes.environments=$lanes"
  done ;;
bc)
  ./run.sh final-bc-cold final final/bc-nolanes.json 12 1 cold:final-bc ;;
memory)
  ./run.sh final-ppo-host final final/ppo.json 6 0 final-ppo 'host_store=true'
  ./run.sh final-ppo-640 final final/ppo.json 6 0 final-ppo-640 'learner.micro_batch_rows=640' ;;
laptop)
  for stage in bc ppo; do for rows in 1280 640 320; do
    EXTRA=memory_fraction=0.49 ./run.sh laptop-$stage-$rows final final2048-$rows/$stage.json 2 0 laptop-$stage-$rows
    grep -q '"updates"' logs/laptop-$stage-$rows.json && break  # the largest that fits
  done; done ;;
esac; done
