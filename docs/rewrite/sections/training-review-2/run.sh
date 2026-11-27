#!/bin/bash
# One GPU job as a systemd user unit under the GPU lock (queue rules), output to DATA/logs/UNIT.log.
#   docs/rewrite/sections/training-review-2/run.sh UNIT script.py [args...]
set -e
unit="bnb-4-$1"; shift
worktree=/home/vizia/avesnova/boost-and-broadside/v2-review4
data=/home/vizia/avesnova/boost-and-broadside/data
systemctl --user reset-failed "$unit" 2>/dev/null || true
mkdir -p "$data/review4/logs"
systemd-run --user --unit "$unit" -p MemoryMax=8G -p RuntimeMaxSec=1800 --working-directory="$worktree" \
  --setenv=PYTHONPATH=. --setenv=PATH="$PATH" \
  -p StandardOutput=append:"$data/review4/logs/$unit.log" -p StandardError=append:"$data/review4/logs/$unit.log" \
  flock "$data/gpu.lock" uv run --no-sync python "docs/rewrite/sections/training-review-2/$@"
