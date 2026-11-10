#!/bin/bash
# Round 16 (the author, 21:00 KST): win rates at fixed odds for a paper, replacing the crossover
# search: the champion and the nearest model with n ships against the scripted agent with n x 1,
# 1.25, 1.5, 2, deathmatch in the training arena, at n = 10, 20, 40 (n = 5's 5:5 .. 5:10 grid is
# already measured); 128 games per ratio; one compile per case.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-campaign TRITON_CACHE_DIR=/tmp/inductor-campaign/triton
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
C=runs/2026-11-09-rl-deadreckon-split-hiddenphysics/policies/step_180355072/uniform_average
M=runs/2026-11-10-rl-nearest/policies/step_40894464/uniform_average
OUT=runs/crossover/fixed-odds
odds() {  # tag, agent, n, block environments, batches
  say "odds start $1"
  if .venv/bin/python docs/rewrite/sections/gpu-box/fixed_odds.py $2 deathmatch_$3v$3_arena $3 \
      $OUT/$1.json --block-environments $4 --batches $5 >> runs/fixed-odds.out 2>&1; then
    say "odds done $1: $(tail -1 runs/fixed-odds.out)"
  else
    say "odds FAILED $1 (runs/fixed-odds.out)"
  fi
}
odds champion-10 $C 10 64 1
odds nearest-10 $M 10 64 1
odds champion-20 $C 20 32 2
odds nearest-20 $M 20 32 2
odds champion-40 $C 40 16 4
odds nearest-40 $M 40 16 4
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
say "odds all done"
