#!/bin/bash
# arenas.sh DIR [GAMES]: DIR/candidates.json at 5v5 at R 3.9 / 15.6 km (arena scale 0.5 / 2
# on the 7.8 km 5v5 default; energy v turn) and 1v1 at R 3.9 / 15.6 km (the density rule's
# 1v1 radius is 7.8 km x sqrt(1/5) = 3.488 km: scale 1.118 / 4.472; with both mirrors)
dir=$1; games=${2:-512}
cd /workspace/ships2/repo
export PYTHONPATH=src:. PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=3 TRITON_DEFAULT_FP_FUSION=1 \
  TORCHINDUCTOR_CACHE_DIR=/workspace/ships2/cache/inductor TRITON_CACHE_DIR=/workspace/ships2/cache/triton
run() {
  tag=$1; size=$2; scale=$3; shift 3
  /workspace/arch/venv/bin/python docs/rewrite/sections/maneuvers-2/sizes.py \
    --candidates ../ships-r9m2/$dir/candidates.json --tag ../ships-r9m2/$dir/$tag \
    --sizes $size --arena-scale $scale --games $games --lineups "$@"
}
run five-small 5 0.5 "energy v turn"
run five-large 5 2.0 "energy v turn"
run one-small 1 1.118 "energy v turn" "energy mirror" "turn mirror"
run one-large 1 4.472 "energy v turn" "energy mirror" "turn mirror"
