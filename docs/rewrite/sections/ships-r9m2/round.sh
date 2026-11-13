#!/bin/bash
# round.sh DIR GAMES: DIR/candidates.json at 1v1 (energy v turn and both mirrors) and 10v10 on
# the density rule, then 5v5 at map radius 3.9 / 7.8 / 15.6 km (arena scale 0.5 / 1 / 2)
dir=$1; games=${2:-512}
cd /workspace/ships2/repo
export PYTHONPATH=src:. PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=3 TRITON_DEFAULT_FP_FUSION=1 \
  TORCHINDUCTOR_CACHE_DIR=/workspace/ships2/cache/inductor TRITON_CACHE_DIR=/workspace/ships2/cache/triton
sizes() {
  /workspace/arch/venv/bin/python docs/rewrite/sections/maneuvers-2/sizes.py \
    --candidates ../ships-r9m2/$dir/candidates.json --games $games "$@"
}
sizes --tag ../ships-r9m2/$dir/sizes --sizes 1 10 --mirror-sizes 1 \
  --lineups "energy v turn" "energy mirror" "turn mirror"
for scale in 0.5 1.0 2.0; do
  sizes --tag ../ships-r9m2/$dir/s5-r$scale --sizes 5 --arena-scale $scale --lineups "energy v turn"
done
