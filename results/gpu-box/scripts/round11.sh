#!/bin/bash
# Round 11 (the author, 10:35 KST): after tournament 7, both RL runs resumed to 172 updates
# (180.4M steps; a resume continues to the run's total), the null then the champion.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
until grep -q "tournament7 done\|tournament7 FAILED with 8 x 128" "$LOG"; do sleep 30; done
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
for name in 2026-11-09-rl-absolute 2026-11-09-rl-deadreckon-split-hiddenphysics; do
  if [ -d runs/$name/policies/step_180355072 ]; then say "$name already at 180.4M"; continue; fi
  say "start resume $name to 172 updates (180.4M)"
  if .venv/bin/bnb train --resume $name --updates 172 >> runs/$name.out 2>&1; then
    say "done resume $name to 180.4M"
  else
    say "FAILED resume $name to 180.4M (runs/$name.out)"
  fi
  docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
done
say "round11 done"
