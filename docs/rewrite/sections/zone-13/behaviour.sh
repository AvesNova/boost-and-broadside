#!/bin/bash
# Item 13: the behaviour conditions (yardstick mirrors 1v1 and 4v4, fields and fog on) on this
# tree, under the GPU lock (one unit).   behaviour.sh OUT
set -u
OUT=$1
TREE=/home/vizia/avesnova/boost-and-broadside/v2-zone13
cd "$TREE"
echo "start $(date +%H:%M:%S) power $(powerprofilesctl get) load $(cut -d' ' -f1-3 /proc/loadavg) tree $(git rev-parse --short HEAD)"
for size in 1 4; do
  case $size in 1) envs=2048;; 4) envs=1024;; esac
  [ -f "$OUT/yardstick_${size}v${size}.json" ] && continue
  PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/zone-13/behaviour.py --agent yardstick --size $size --envs $envs --out "$OUT" || echo "FAILED $size"
done
echo "done $(date +%H:%M:%S)"
