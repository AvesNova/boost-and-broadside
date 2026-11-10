#!/bin/bash
# Fleet ratings, cut to size (the author, 20:50 KST): champion, nearest model, scripted only; 20v20
# reusing its earlier games, then 40v40; training arena. (round15.sh's copy listed a --reuse
# directory that does not exist yet and failed at once.)
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
C=runs/2026-11-09-rl-deadreckon-split-hiddenphysics/policies/step_180355072/uniform_average
M=runs/2026-11-10-rl-nearest/policies/step_40894464/uniform_average
ratings() {  # name, block environments, presets...
  local name=$1 envs=$2; shift 2
  rm -rf runs/ratings/fleet3-$name
  say "fleet3 $name start (champion, nearest, scripted; 8 x $envs)"
  if .venv/bin/bnb ratings 2026-11-09-bc-absolute --no-league --no-ladder --with $C $M \
      --presets "$@" --target-stderr 35 --budget "${BUDGET:-20480}" --blocks 8 --block-environments $envs \
      --reuse runs/ratings/fleet-20v20-arena --out runs/ratings/fleet3-$name \
      >> runs/ratings-fleet3-$name.out 2>&1; then
    say "fleet3 $name done"
  else
    say "fleet3 $name FAILED (runs/ratings-fleet3-$name.out)"
  fi
}
ratings 20v20-arena 128 frontline_20v20_arena deathmatch_20v20_arena
ratings 40v40-arena 64 frontline_40v40_arena deathmatch_40v40_arena
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
say "fleet3 done"
