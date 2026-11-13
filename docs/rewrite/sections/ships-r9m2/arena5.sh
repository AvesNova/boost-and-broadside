#!/bin/bash
# arena5.sh DIR GAMES SIZE SCALE...: DIR/candidates.json, energy v turn at SIZE at each arena
# scale (on the density rule's radius: 7.8 km at 5v5)
dir=$1; games=$2; size=$3; shift 3
cd /workspace/ships2/repo
export PYTHONPATH=src:. PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=3 TRITON_DEFAULT_FP_FUSION=1 \
  TORCHINDUCTOR_CACHE_DIR=/workspace/ships2/cache/inductor TRITON_CACHE_DIR=/workspace/ships2/cache/triton
for scale in "$@"; do
  /workspace/arch/venv/bin/python docs/rewrite/sections/maneuvers-2/sizes.py \
    --candidates ../ships-r9m2/$dir/candidates.json --tag ../ships-r9m2/$dir/s${size}-r$scale \
    --sizes $size --arena-scale $scale --games $games --lineups "energy v turn"
done
