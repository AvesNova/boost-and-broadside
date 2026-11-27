#!/bin/bash
# One GPU job as a systemd user unit under the GPU lock (queue rules), output to DATA/logs/UNIT.log.
#   docs/rewrite/sections/teacher-6/run.sh UNIT script.py [args...]   (run from the worktree)
set -e
unit="bnb-6-$1"; shift
worktree=${WORKTREE:-/home/vizia/avesnova/boost-and-broadside/v2-teacher6}
data=/home/vizia/avesnova/boost-and-broadside/data
systemctl --user reset-failed "$unit" 2>/dev/null || true
mkdir -p "$data/teacher6/logs" "$data/teacher6/tmp"
systemd-run --user --unit "$unit" -p MemoryMax=12G -p RuntimeMaxSec=3600 --working-directory="$worktree" \
  --setenv=PYTHONPATH=. --setenv=PATH="$PATH" --setenv=TMPDIR="$data/teacher6/tmp" \
  -p StandardOutput=append:"$data/teacher6/logs/$unit.log" -p StandardError=append:"$data/teacher6/logs/$unit.log" \
  flock "$data/gpu.lock" uv run --no-sync python "$@"
