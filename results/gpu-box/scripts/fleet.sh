#!/bin/bash
# Fleet-size ratings (the author, 17:20 KST): zero-shot Elo at 5v5, 10v10, 20v20, 40v40, each
# larger size both in the training arena (2600 m, `_arena`) and at the training density
# (`_density`). One field per size and variant: the scripted agent (1000), the stationary ladder
# below it (scripted at higher temperatures), the null's RL 20/40/80/180M uniform averages above
# it (calibration rungs), and each agent's best checkpoint (champion and null RL 180.4M uniform,
# nearest RL 40.9M uniform). 5v5 reuses tournament 10's games. Target +-35, at most 20480 games
# per preset; the layout shrinks with the ships per game.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
N=runs/2026-11-09-rl-absolute/policies
players=($N/step_20971520/uniform_average $N/step_40894464/uniform_average
  $N/step_80740352/uniform_average $N/step_180355072/uniform_average
  runs/2026-11-09-rl-deadreckon-split-hiddenphysics/policies/step_180355072/uniform_average
  runs/2026-11-10-rl-nearest/policies/step_40894464/uniform_average)
rate() {  # name, block environments, presets...
  local name=$1 envs=$2; shift 2
  say "fleet $name start (${#players[@]} players + scripted + ladder; 8 x $envs)"
  if .venv/bin/bnb ratings 2026-11-09-bc-absolute --no-league --with "${players[@]}" \
      --presets "$@" --target-stderr 35 --budget 20480 --blocks 8 --block-environments $envs \
      --reuse runs/ratings/round10 --out runs/ratings/fleet-$name >> runs/ratings-fleet-$name.out 2>&1; then
    say "fleet $name done"
  else
    say "fleet $name FAILED (runs/ratings-fleet-$name.out)"
  fi
  docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
}
rate 5v5 256 frontline deathmatch
rate 10v10-arena 256 frontline_10v10_arena deathmatch_10v10_arena
rate 10v10-density 256 frontline_10v10_density deathmatch_10v10_density
rate 20v20-arena 128 frontline_20v20_arena deathmatch_20v20_arena
rate 20v20-density 128 frontline_20v20_density deathmatch_20v20_density
rate 40v40-arena 64 frontline_40v40_arena deathmatch_40v40_arena
rate 40v40-density 64 frontline_40v40_density deathmatch_40v40_density
say "fleet done"
