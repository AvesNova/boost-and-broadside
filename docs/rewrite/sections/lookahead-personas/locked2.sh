#!/bin/bash
# One GPU job under the shared lock, in its own memory-capped user unit (round 2):
#   locked2.sh NAME COMMAND...   (log: DATA/logs/NAME.log, appended)
set -u
name=$1; shift
tree=/home/vizia/avesnova/boost-and-broadside/v2-personas2
data=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2
systemctl --user reset-failed "bnb-p2-$name" 2>/dev/null
flock /home/vizia/avesnova/boost-and-broadside/data/gpu.lock \
  systemd-run --user --wait --collect --unit "bnb-p2-$name" -p MemoryMax=12G \
  --working-directory "$tree" -E PYTHONPATH="$tree" -E PATH="$PATH" \
  bash -c "echo \"== \$(date +%H:%M:%S) start\" >> $data/logs/$name.log; { $*; } >> $data/logs/$name.log 2>&1; echo \"== \$(date +%H:%M:%S) exit \$?\" >> $data/logs/$name.log"
