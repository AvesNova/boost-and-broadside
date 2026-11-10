#!/bin/bash
# Tournament 8 (gate 3 of the nearest A/B): after the nearest BC reaches 120M, rate its 120M
# live / uniform / exponential and its 30M and 60M uniform against the champion's and the null's
# BC uniform averages at 30M, 60M and 120M, in tournament 7's field scale (reusing its games and
# tournaments 5-6's), scripted = 1000. 8 x 256, target +-35.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
until grep -q "done 2026-11-10-bc-nearest\|FAILED 2026-11-10-bc-nearest" "$LOG"; do sleep 20; done
grep -q "done 2026-11-10-bc-nearest" "$LOG" || { say "tournament8 not started: the nearest BC failed"; exit 1; }
N=runs/2026-11-10-bc-nearest/policies
last=$(ls -d $N/step_* | sort -V | tail -1)
players=($last/live $last/uniform_average $last/exponential_average
  $N/step_30408704/uniform_average $N/step_60817408/uniform_average)
for line in deadreckon-split-hiddenphysics absolute; do
  for s in 30408704 60817408 120586240; do players+=(runs/2026-11-09-bc-$line/policies/step_$s/uniform_average); done
done
say "tournament8 start (${#players[@]} players: nearest BC vs the champion and null BC curves)"
if .venv/bin/bnb ratings 2026-11-09-bc-absolute --no-league --no-ladder --with "${players[@]}" \
    --presets frontline deathmatch --target-stderr 35 --blocks 8 --block-environments 256 \
    --reuse runs/ratings/round5 runs/ratings/round6 runs/ratings/round7-8x256 \
    --out runs/ratings/round8 >> runs/ratings-round8.out 2>&1; then
  say "tournament8 done"
else
  say "tournament8 FAILED (runs/ratings-round8.out)"
fi
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
