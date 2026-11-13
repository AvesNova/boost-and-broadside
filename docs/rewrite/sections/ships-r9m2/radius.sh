#!/bin/bash
# radius.sh DIR GAMES [SCALES...]: DIR/candidates.json, energy v turn at 5v5 at each arena
# scale (default 0.5 1.0 2.0: R 3.9 / 7.8 / 15.6 km)
dir=$1; games=${2:-512}; shift 2; scales=${@:-0.5 1.0 2.0}
cd /workspace/ships2/repo
export PYTHONPATH=src:. PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=3 TRITON_DEFAULT_FP_FUSION=1 \
  TORCHINDUCTOR_CACHE_DIR=/workspace/ships2/cache/inductor TRITON_CACHE_DIR=/workspace/ships2/cache/triton
for scale in $scales; do
  /workspace/arch/venv/bin/python docs/rewrite/sections/maneuvers-2/sizes.py \
    --candidates ../ships-r9m2/$dir/candidates.json --games $games \
    --tag ../ships-r9m2/$dir/s5-r$scale --sizes 5 --arena-scale $scale --lineups "energy v turn"
done
