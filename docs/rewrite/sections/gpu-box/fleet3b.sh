#!/bin/bash
# The author's "too slow" (21:40 KST): the champion beat everyone in every 20v20 deathmatch game
# (its Elo only a lower bound, ~2550, the error stuck at +-142), so the 20v20 tournament was stopped;
# its frontline from the reused games (no new games), then 40v40 with 4096 games per mode at most.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
C=runs/2026-11-09-rl-deadreckon-split-hiddenphysics/policies/step_180355072/uniform_average
M=runs/2026-11-10-rl-nearest/policies/step_40894464/uniform_average
go() {  # name, envs, budget, reuse dirs (comma), presets...
  local name=$1 envs=$2 budget=$3 reuse=$4; shift 4
  say "fleet3 $name start (budget $budget)"
  if .venv/bin/bnb ratings 2026-11-09-bc-absolute --no-league --no-ladder --with $C $M \
      --presets "$@" --target-stderr 35 --budget $budget --blocks 8 --block-environments $envs \
      --reuse ${reuse//,/ } --out runs/ratings/fleet3-$name >> runs/ratings-fleet3-$name.out 2>&1; then
    say "fleet3 $name done"
  else
    say "fleet3 $name FAILED (runs/ratings-fleet3-$name.out)"
  fi
}
go 20v20-frontline 128 2048 runs/ratings/fleet-20v20-arena frontline_20v20_arena
go 40v40-arena 64 4096 runs/ratings/fleet-20v20-arena frontline_40v40_arena deathmatch_40v40_arena
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
say "fleet3 all done"
