#!/bin/bash
# The A/B tournament on the champion's code (both arms trained on this game): the one-head arm
# and the champion at matched checkpoints, the BC 120.6M uniform average and the RL uniform
# average and live policy at 60.8M, 121.6M and 180.4M, the null's RL 180.4M uniform average
# for context, and the scripted agent (the 1000 anchor). Frontline and deathmatch rated
# separately; A-optimal allocation to a worst standard error of 12 (budget 80,000 games a
# preset), as the temperature sweep. Resumable: `bnb ratings` goes on from its records.
set -u
cd /workspace/boost-and-broadside
LOG=runs/pipeline.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
/venv/main/bin/python docs/rewrite/sections/gpubox2/fetch_champion.py runs >> runs/fetch.out 2>&1 \
  || { say "tournament: champion fetch FAILED (runs/fetch.out)"; exit 1; }
players=()
add() { if [ -f "$1/checkpoint.json" ]; then players+=("$1"); else say "tournament: missing $1"; exit 1; fi; }
for pair in "2026-11-12-bc-onehead 2026-11-12-rl-onehead" \
    "2026-11-09-bc-deadreckon-split-hiddenphysics 2026-11-09-rl-deadreckon-split-hiddenphysics"; do
  set -- $pair
  add runs/$1/policies/step_120586240/uniform_average
  for s in 60817408 121634816 180355072; do
    for v in uniform_average live; do add runs/$2/policies/step_$s/$v; done
  done
done
add runs/2026-11-09-rl-absolute/policies/step_180355072/uniform_average
for layout in "8 512" "8 256"; do
  set -- $layout
  out=runs/ratings/onehead-$1x$2
  say "tournament start (${#players[@]} players + scripted; $1 x $2; target 12; $out)"
  if .venv/bin/bnb ratings 2026-11-12-rl-onehead --no-league --no-ladder --with "${players[@]}" \
      --presets frontline deathmatch --budget 80000 --target-stderr 12 \
      --blocks $1 --block-environments $2 --out $out >> runs/ratings-onehead.out 2>&1; then
    say "tournament done ($1 x $2)"; touch runs/tournament.done; exit 0
  fi
  say "tournament FAILED with $1 x $2 (runs/ratings-onehead.out)"
  grep -q "out of memory\|OutOfMemory" runs/ratings-onehead.out || exit 1
done
exit 1
