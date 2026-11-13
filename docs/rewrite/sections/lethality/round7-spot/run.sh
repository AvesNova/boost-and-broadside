#!/bin/bash
# Round 7's spot-checks (game v15): round 6's two closest cards and the base card at
# 10v10, and at 5v5 on 0.5x and 2x arenas, in elimination (180 s), kill-lead deathmatch
# (600 s; read its kill share) and frontline (600 s). Self-contained: run from the
# repository root of a feat/lethality checkout. Writes JSON / MD / logs into
# docs/rewrite/sections/lethality/round7-spot/. Fits an 8 GB GPU (10v10 at B 2048,
# 5v5 at B 4096, as maneuvers-2's laptop runs did); one job at a time.
#
#     bash docs/rewrite/sections/lethality/round7-spot/run.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH=src:.
sizes=docs/rewrite/sections/maneuvers-2/sizes.py
here=docs/rewrite/sections/lethality/round7-spot
rel=../lethality/round7-spot  # sizes.py reads paths relative to maneuvers-2
declare -A clock=([elimination]=180 [deathmatch]=600 [frontline]=600)
for mode in elimination deathmatch frontline; do
  uv run --no-sync python "$sizes" --mode "$mode" --seconds "${clock[$mode]}" \
    --candidates "$rel/candidates.json" --tag "$rel/${mode}-10v10" --sizes 10 --games 256 \
    > "$here/${mode}-10v10.log" 2>&1
  for scale in 0.5 2; do
    uv run --no-sync python "$sizes" --mode "$mode" --seconds "${clock[$mode]}" \
      --candidates "$rel/candidates.json" --tag "$rel/${mode}-arena${scale}" --sizes 5 \
      --games 256 --arena-scale "$scale" > "$here/${mode}-arena${scale}.log" 2>&1
  done
done
echo "done: $here"
