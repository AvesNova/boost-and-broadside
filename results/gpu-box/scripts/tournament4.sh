#!/bin/bash
# Tournament 4, relaunched with a larger engine layout to fill the GPU (the author, 15:00 KST:
# "fewer larger batches"): 16 blocks x 256 games (4096 games, 16 pairs a batch) instead of the
# default 8 x 128; falls back to 16 x 128 if that fails. It goes on from its batch record.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
P=policies/step_30408704/uniform_average
B=2026-11-09-bc
players=(runs/$B-absolute/policies/rewound_1/step_30408704/uniform_average
  runs/$B-absolute-seed1/$P runs/$B-deadreckon-split/$P runs/$B-deadreckon-split-seed1/$P
  runs/$B-deadreckon-split-fourier/$P runs/$B-deadreckon-split-fastdecay-laptop/$P
  runs/$B-persist-split-laptop/$P runs/$B-deadreckon-split-hiddenphysics/$P
  runs/$B-deadreckon-split-hiddenphysics-seed1/$P runs/$B-deadreckon-split-hiddenphysics-tau1.25/$P
  runs/$B-deadreckon-split-hiddenphysics-tau10-laptop/$P runs/$B-deadreckon-split-hiddenphysics-fade/$P
  runs/$B-persist-split-hiddenphysics-laptop/$P)
for layout in "16 256" "16 128"; do
  set -- $layout
  say "tournament4 start (13 players, $1 blocks x $2 games)"
  if .venv/bin/bnb ratings $B-absolute --no-league --no-ladder --with "${players[@]}" \
      --presets frontline deathmatch --blocks $1 --block-environments $2 \
      --out runs/ratings/round4 >> runs/ratings-round4.out 2>&1; then
    say "tournament4 done"; break
  fi
  say "tournament4 FAILED with $1 x $2 (runs/ratings-round4.out)"
done
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
