#!/bin/bash
# One GPU job under the shared lock, in its own memory-capped user unit:
#   locked.sh NAME COMMAND...   (log: DATA/logs/NAME.log, appended)
set -u
name=$1; shift
tree=/home/vizia/avesnova/boost-and-broadside/v2-defaults
data=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas
systemctl --user reset-failed "bnb-persona-$name" 2>/dev/null
flock /home/vizia/avesnova/boost-and-broadside/data/gpu.lock \
  systemd-run --user --wait --collect --unit "bnb-persona-$name" -p MemoryMax=12G \
  --working-directory "$tree" -E PYTHONPATH="$tree" -E PATH="$PATH" \
  bash -c "echo \"== \$(date +%H:%M:%S) start\" >> $data/logs/$name.log; { $*; } >> $data/logs/$name.log 2>&1; echo \"== \$(date +%H:%M:%S) exit \$?\" >> $data/logs/$name.log"
