#!/bin/bash
# Elimination shrink sweep: start (hold) x duration; clock = hold + duration + 8 (the end lead).
cd /home/vizia/avesnova/boost-and-broadside/v2; export PYTHONPATH=src
PY=/home/vizia/avesnova/boost-and-broadside/v2/.venv/bin/python
for hold in 0 18 36 60 90; do for duration in 60 90 135 180; do
  clock=$((hold + duration + 8)); name=h${hold}_d${duration}
  echo "== $name start $(date +%H:%M:%S)"
  timeout 900 $PY -m bench.mode_dynamics elimination --games 256 --batch 128 --fields     --lineup "yongma,jujak;jujak,yongma" --seed 61 --episode-seconds $clock     --set mode.shrink_hold_seconds=$hold --label $name --out docs/rewrite/sections/mode-timing/data/data.jsonl > /tmp/claude-1000/-home-vizia-avesnova-boost-and-broadside-v2/bd37d180-77ce-450e-aae2-7252ffa0d40a/scratchpad/elim-shrink/$name.log 2>&1
  echo "== $name done $(date +%H:%M:%S) rc=$?"
done; done
