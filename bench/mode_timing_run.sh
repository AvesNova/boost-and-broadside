#!/bin/bash
# The mode-timing measurements, one at a time on the CPU (eager), each bounded to 40 minutes.
# Usage (from the repo root): PYTHON=<python> bash bench/mode_timing_run.sh
# Per-batch results append to docs/rewrite/sections/mode-timing/data/<name>.jsonl.
OUT=docs/rewrite/sections/mode-timing/data
mkdir -p $OUT
export PYTHONPATH=src OMP_NUM_THREADS=16
PY=${PYTHON:-python}
run() {
  name=$1; shift
  echo "== $name $(date +%T)"
  timeout 2400 $PY -m bench.mode_dynamics "$@" --out $OUT/$name.jsonl --label $name \
    --eager --device cpu --fields --lineup "yongma,jujak;jujak,yongma"
  echo "== $name done $(date +%T) rc=$?"
}
run deathmatch deathmatch --games 80 --batch 16 --episode-seconds 900
run elimination_noshrink elimination --games 128 --batch 32 --episode-seconds 600 \
  --set mode.shrink_hold_seconds=100000
run elimination_shrink elimination --games 128 --batch 32
run frontline_2v2 frontline --games 128 --batch 32 --ships 2 --density
run frontline_3v3 frontline --games 128 --batch 32 --ships 3 --density
