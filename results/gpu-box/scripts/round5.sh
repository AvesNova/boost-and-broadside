#!/bin/bash
# Round 5: tournament 3's front runners to 60M (the author's rule: every arm the tournament
# cannot separate). Box: deadreckon-split-hiddenphysics, deadreckon-split, resumed from their
# 30.4M checkpoints with the budget raised (every other setting from the checkpoint). The
# laptop resumes persist-split-laptop. Then tournament 4 at 60M (tournament4.sh).
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
for name in 2026-11-09-bc-deadreckon-split-hiddenphysics 2026-11-09-bc-deadreckon-split; do
  say "start resume $name to 60M"
  if .venv/bin/bnb train --resume $name --bc-budget-steps 60000000 >> runs/$name.out 2>&1; then
    say "done resume $name"
  else
    say "FAILED resume $name (runs/$name.out)"
  fi
  docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
done
say "round5 box done"
