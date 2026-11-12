#!/bin/bash
# The respawn-wave runs on the box: one process per variant set (each compiles once).
# usage: run.sh NAME KIND SHIPS RADIUS SLOTS PER_SLOT GAMES EPISODE_SECONDS
set -u
root=${WAVES_ROOT:-/workspace/waves2}
cd $root/bnb
name=$1
out=$root/results/respawn-waves
mkdir -p $out
export TRITON_DEFAULT_FP_FUSION=1 TORCHINDUCTOR_COMPILE_THREADS=3 PYTHONPATH=src PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=$root/cache/inductor TRITON_CACHE_DIR=$root/cache/triton
.venv/bin/python -m bench.frontline_design --kind $2 \
  --variants docs/rewrite/sections/respawn-waves/variants/$name.json \
  --out $out/$name.jsonl \
  --ships $3 --radius $4 --slots $5 --per-slot $6 --games $7 --episode-seconds $8 \
  2>&1 | tee -a $out/$name.log
