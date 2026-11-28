#!/bin/bash
# One GPU job as a systemd user unit under the GPU lock (queue rules), output to DATA/gradients11/logs/UNIT.log.
#   docs/rewrite/sections/gradients-11/run.sh UNIT script-or-module args...   (python args)
set -e
unit="bnb-11-$1"; shift
worktree=/home/vizia/avesnova/boost-and-broadside/v2-gradients11
data=/home/vizia/avesnova/boost-and-broadside/data
systemctl --user reset-failed "$unit" 2>/dev/null || true
mkdir -p "$data/gradients11/logs" "$data/gradients11/tmp"
echo "$(date +%H:%M:%S) start $unit load $(cut -d' ' -f1 /proc/loadavg) $(powerprofilesctl get)" >> "$data/gradients11/logs/$unit.log"
systemd-run --user --unit "$unit" -p MemoryMax=14G -p RuntimeMaxSec=7200 --working-directory="$worktree" \
  --setenv=PYTHONPATH=. --setenv=PATH="$PATH" --setenv=TMPDIR="$data/gradients11/tmp" \
  -p StandardOutput=append:"$data/gradients11/logs/$unit.log" -p StandardError=append:"$data/gradients11/logs/$unit.log" \
  flock "$data/gpu.lock" uv run --no-sync python "$@"
