#!/bin/bash
# The author (20:35 KST): drop the map-size deviation (the `_density` presets). After the running
# 20v20 arena ratings, 40v40 in the training arena only; the crossover campaign restarts without
# the density presets once its running case (nearest, frontline 10v10 arena) is done.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
(
  until grep -q "crossover done nearest frontline_10v10_arena\|crossover FAILED nearest frontline_10v10_arena" "$LOG"; do sleep 15; done
  runs/killpat.sh "crossover_campaign\.p[y]"
  say "crossover campaign restarted without the density presets"
  TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-campaign TRITON_CACHE_DIR=/tmp/inductor-campaign/triton \
    python3 docs/rewrite/sections/gpu-box/crossover_campaign.py > runs/crossover-campaign.nohup 2>&1
) &
while pgrep -f "fleet-20v20-arena" > /dev/null; do sleep 30; done
say "fleet 20v20-arena done"
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
N=runs/2026-11-09-rl-absolute/policies
players=($N/step_20971520/uniform_average $N/step_40894464/uniform_average
  $N/step_80740352/uniform_average $N/step_180355072/uniform_average
  runs/2026-11-09-rl-deadreckon-split-hiddenphysics/policies/step_180355072/uniform_average
  runs/2026-11-10-rl-nearest/policies/step_40894464/uniform_average)
say "fleet 40v40-arena start (${#players[@]} players + scripted; 8 x 64)"
if .venv/bin/bnb ratings 2026-11-09-bc-absolute --no-league --no-ladder --with "${players[@]}" \
    --presets frontline_40v40_arena deathmatch_40v40_arena --target-stderr 35 --budget 20480 \
    --blocks 8 --block-environments 64 --out runs/ratings/fleet-40v40-arena \
    >> runs/ratings-fleet-40v40-arena.out 2>&1; then
  say "fleet 40v40-arena done"
else
  say "fleet 40v40-arena FAILED (runs/ratings-fleet-40v40-arena.out)"
fi
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
say "fleet done"
wait
