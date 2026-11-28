#!/bin/bash
# Item 12: the behaviour conditions for one tree (core's rules or the new): centre and yardstick
# mirrors at 1v1 / 4v4 / 16v16. Run from the tree to measure; under the GPU lock (one unit).
#   behaviour.sh TREE OUT
set -u
TREE=$1; OUT=$2
SCRIPT=/home/vizia/avesnova/boost-and-broadside/v2-rules12/docs/rewrite/sections/rules-12/behaviour.py
export UV_PROJECT_ENVIRONMENT=/home/vizia/avesnova/boost-and-broadside/v2-rules12/.venv
cd "$TREE"
echo "start $(date +%H:%M:%S) power $(powerprofilesctl get) load $(cut -d' ' -f1-3 /proc/loadavg) tree $(git rev-parse --short HEAD)"
for size in 1 4 16; do
  case $size in 1) envs=4096;; 4) envs=2048;; 16) envs=512;; esac
  for agent in centre yardstick; do
    [ -f "$OUT/${agent}_${size}v${size}.json" ] && continue
    PYTHONPATH=. uv run --no-sync python "$SCRIPT" --agent $agent --size $size --envs $envs --out "$OUT" || echo "FAILED $agent $size"
  done
done
echo "done $(date +%H:%M:%S)"
