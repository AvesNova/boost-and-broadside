#!/bin/bash
# One GPU job under the shared lock, in its own memory-capped user unit (round 3):
#   locked3.sh NAME COMMAND...   (log: DATA/logs/NAME.log, appended)
set -u
name=$1; shift
tree=/home/vizia/avesnova/boost-and-broadside/v2-personas3
data=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-3
systemctl --user reset-failed "bnb-p3-$name" 2>/dev/null
flock /home/vizia/avesnova/boost-and-broadside/data/gpu.lock \
  systemd-run --user --wait --collect --unit "bnb-p3-$name" -p MemoryMax=12G \
  --working-directory "$tree" -E PYTHONPATH="$tree" -E PATH="$PATH" \
  bash -c "echo \"== \$(date +%H:%M:%S) start\" >> $data/logs/$name.log; { $*; } >> $data/logs/$name.log 2>&1; echo \"== \$(date +%H:%M:%S) exit \$?\" >> $data/logs/$name.log"
