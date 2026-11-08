#!/bin/bash
# Tournament 1 (round 2 on the 3090): five 30M uniform averages and the scripted agent
# (1000), frontline then deathmatch, no ladder (with it: 14 players, 91 pairs, too slow).
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=policies/step_30408704/uniform_average
echo "$(date -u +%H:%M) tournament1 start (no ladder)" >> runs/arms.log
if .venv/bin/bnb ratings 2026-11-08-bc-arm-C-fixed-3090 --no-league --no-ladder \
    --with runs/2026-11-08-bc-arm-C-fixed-3090/$P runs/2026-11-08-bc-arm2d-persistence-3090/$P \
      runs/2026-11-08-bc-arm3d-split-3090/$P runs/2026-11-08-bc-arm4d-extrapolation-3090/$P \
      runs/2026-11-08-bc-arm-C-fixed-LB-3090/$P \
    --presets frontline deathmatch --out runs/ratings/round2 >> runs/ratings-round2.out 2>&1; then
  echo "$(date -u +%H:%M) tournament1 done" >> runs/arms.log
else
  echo "$(date -u +%H:%M) tournament1 FAILED (runs/ratings-round2.out)" >> runs/arms.log
fi
