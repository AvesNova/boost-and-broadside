#!/bin/bash
# Tournament 3: round 3 redone on C-fixed's base. Every finished 30M uniform average and the
# scripted agent (1000), frontline then deathmatch, no ladder. A missing arm is skipped and named.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
until grep -q "round4b ready for the tournament" "$LOG"; do sleep 30; done
P=policies/step_30408704/uniform_average
players=()
for run in 2026-11-09-bc-absolute 2026-11-09-bc-absolute-seed1 2026-11-09-bc-deadreckon-split \
    2026-11-09-bc-deadreckon-split-fourier 2026-11-09-bc-deadreckon-split-hiddenphysics \
    2026-11-09-bc-deadreckon-split-fastdecay-laptop 2026-11-09-bc-persist-split-laptop; do
  if [ -f "runs/$run/$P/checkpoint.json" ]; then players+=("runs/$run/$P"); else say "tournament3: $run missing, skipped"; fi
done
say "tournament3 start (${#players[@]} players)"
if .venv/bin/bnb ratings 2026-11-09-bc-absolute --no-league --no-ladder --with "${players[@]}" \
    --presets frontline deathmatch --out runs/ratings/round3 >> runs/ratings-round3.out 2>&1; then
  say "tournament3 done"
else
  say "tournament3 FAILED (runs/ratings-round3.out)"
fi
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
