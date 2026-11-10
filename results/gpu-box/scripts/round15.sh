#!/bin/bash
# Round 15 (the author, 20:50 KST): the evaluation cut to size. Crossover: deathmatch in the
# training arena only, the champion and the nearest model only, against the scripted agent, by the
# adaptive search (`crossover_adaptive.py`: one compile per case, the previous fleet size's ratio
# as the first guess, a logistic fit). Fleet ratings: the champion, the nearest model and the
# scripted agent only (no null, no calibration rungs), 20v20 (reusing its games) and 40v40, in
# the training arena.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
C=runs/2026-11-09-rl-deadreckon-split-hiddenphysics/policies/step_180355072/uniform_average
M=runs/2026-11-10-rl-nearest/policies/step_40894464/uniform_average
OUT=runs/crossover/adaptive

ratings() {  # name, block environments, presets...
  local name=$1 envs=$2; shift 2
  say "fleet3 $name start (champion, nearest, scripted; 8 x $envs)"
  if .venv/bin/bnb ratings 2026-11-09-bc-absolute --no-league --no-ladder --with $C $M \
      --presets "$@" --target-stderr 35 --budget 20480 --blocks 8 --block-environments $envs \
      --reuse runs/ratings/fleet-5v5 runs/ratings/fleet-20v20-arena runs/ratings/fleet-40v40-arena \
      --out runs/ratings/fleet3-$name >> runs/ratings-fleet3-$name.out 2>&1; then
    say "fleet3 $name done"
  else
    say "fleet3 $name FAILED (runs/ratings-fleet3-$name.out)"
  fi
}
(
  ratings 20v20-arena 128 frontline_20v20_arena deathmatch_20v20_arena
  ratings 40v40-arena 64 frontline_40v40_arena deathmatch_40v40_arena
  docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
  say "fleet3 done"
) &

ratio() { python3 -c "import json;print(json.load(open('$1'))['crossing']['ratio'])"; }
cross() {  # tag, agent, preset, n, prior ratio, block environments, batches
  local tag=$1
  say "crossover2 start $tag (prior x$5)"
  if TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-campaign TRITON_CACHE_DIR=/tmp/inductor-campaign/triton \
      .venv/bin/python docs/rewrite/sections/gpu-box/crossover_adaptive.py $2 $3 $4 $5 $OUT/$tag.json \
      --block-environments $6 --batches $7 >> runs/crossover-adaptive.out 2>&1; then
    say "crossover2 done $tag: $(tail -1 runs/crossover-adaptive.out)"
  else
    say "crossover2 FAILED $tag (runs/crossover-adaptive.out)"
  fi
}
cross champion-dm10 $C deathmatch_10v10_arena 10 1.49 64 3
cross nearest-dm10 $M deathmatch_10v10_arena 10 1.13 64 3
cross champion-dm20 $C deathmatch_20v20_arena 20 $(ratio $OUT/champion-dm10.json) 32 3
cross nearest-dm20 $M deathmatch_20v20_arena 20 $(ratio $OUT/nearest-dm10.json) 32 3
cross champion-dm40 $C deathmatch_40v40_arena 40 $(ratio $OUT/champion-dm20.json) 16 4
cross nearest-dm40 $M deathmatch_40v40_arena 40 $(ratio $OUT/nearest-dm20.json) 16 4
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
say "crossover2 done"
wait
