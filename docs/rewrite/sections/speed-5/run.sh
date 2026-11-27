#!/bin/bash
# One GPU job as a systemd user unit under the GPU lock (queue rules), output to DATA/logs/UNIT.log.
#   docs/rewrite/sections/speed-5/run.sh UNIT script.py [args...]   (run from the worktree)
set -e
unit="bnb-5-$1"; shift
worktree=${WORKTREE:-/home/vizia/avesnova/boost-and-broadside/v2-speed5}
data=/home/vizia/avesnova/boost-and-broadside/data
systemctl --user reset-failed "$unit" 2>/dev/null || true
mkdir -p "$data/speed5/logs" "$data/speed5/tmp"
systemd-run --user --unit "$unit" -p MemoryMax=12G -p RuntimeMaxSec=3600 --working-directory="$worktree" \
  --setenv=PYTHONPATH=. --setenv=PATH="$PATH" --setenv=TMPDIR="$data/speed5/tmp" \
  -p StandardOutput=append:"$data/speed5/logs/$unit.log" -p StandardError=append:"$data/speed5/logs/$unit.log" \
  flock "$data/gpu.lock" uv run --no-sync python "$@"
