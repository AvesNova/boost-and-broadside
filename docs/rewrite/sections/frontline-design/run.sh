#!/bin/bash
# The design search on the box: one process per fleet size (each compiles once).
# usage: run.sh SHIPS RADIUS SLOTS PER_SLOT GAMES [VARIANTS (default vSHIPS)]
set -u
cd /workspace/bnb
n=$1
name=${6:-v$n}
mkdir -p /workspace/results/frontline-design
export TRITON_DEFAULT_FP_FUSION=1 TORCHINDUCTOR_COMPILE_THREADS=6 PYTHONPATH=src
.venv/bin/python -m bench.frontline_design \
  --variants docs/rewrite/sections/frontline-design/variants/$name.json \
  --out /workspace/results/frontline-design/$name.jsonl \
  --ships $n --radius $2 --slots $3 --per-slot $4 --games $5 \
  2>&1 | tee -a /workspace/results/frontline-design/$name.log
