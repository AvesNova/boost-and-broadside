#!/bin/bash
# Tournament 9: the true exponential averages (the snapshot bug, d62bbc1d), the RL 180.4M points
# and the nearest model's exponential, in tournament 7's scale (reusing tournaments 5-8's
# games), scripted = 1000. 8 x 256, target +-35. Decides the nearest RL's starting policy.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
N=runs/2026-11-10-bc-nearest/policies/step_120586240
players=($N/live $N/exponential_average)
for line in deadreckon-split-hiddenphysics absolute; do
  B=runs/2026-11-09-bc-$line/policies/step_120586240
  R=runs/2026-11-09-rl-$line/policies
  players+=($B/uniform_average $B/exponential_average $B/live)
  for s in 60817408 121634816; do players+=($R/step_$s/uniform_average $R/step_$s/exponential_average); done
  players+=($R/step_180355072/live $R/step_180355072/uniform_average $R/step_180355072/exponential_average)
done
say "tournament9 start (${#players[@]} players: true exponentials, RL 180.4M, nearest)"
if .venv/bin/bnb ratings 2026-11-09-bc-absolute --no-league --no-ladder --with "${players[@]}" \
    --presets frontline deathmatch --target-stderr 35 --blocks 8 --block-environments 256 \
    --reuse runs/ratings/round5 runs/ratings/round6 runs/ratings/round7-8x256 runs/ratings/round8 \
    --out runs/ratings/round9 >> runs/ratings-round9.out 2>&1; then
  say "tournament9 done"
else
  say "tournament9 FAILED (runs/ratings-round9.out)"
fi
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
