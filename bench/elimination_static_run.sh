#!/bin/bash
# Elimination at fixed map sizes (no shrink): game length and deaths per minute by radius.
# Spawns, fields and the boundary ramp scale with the radius (layout map_scale).
cd /home/vizia/avesnova/boost-and-broadside-v2; export PYTHONPATH=src
PY=/home/vizia/avesnova/boost-and-broadside-v2/.venv/bin/python
while systemctl --user is-active -q elim-shrink; do sleep 15; done
for radius in 7800 6000 4500 3500 2500 1800 1200 800 500 300 100; do
  name=r$radius; echo "== $name start $(date +%H:%M:%S)"
  timeout 1200 $PY -m bench.mode_dynamics elimination --games 256 --batch 128 --fields     --lineup "yongma,jujak;jujak,yongma" --seed 61 --episode-seconds 900 --radius $radius     --set mode.shrink_hold_seconds=100000 --label $name --out docs/rewrite/sections/mode-timing/data/data.jsonl > /tmp/claude-1000/-home-vizia-avesnova-boost-and-broadside-v2/bd37d180-77ce-450e-aae2-7252ffa0d40a/scratchpad/elim-static/$name.log 2>&1
  echo "== $name done $(date +%H:%M:%S) rc=$?"
done
