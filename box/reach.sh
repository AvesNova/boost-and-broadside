#!/bin/bash
# Laptop side: every minute, whether the box's SSH ports accept a connection (direct, proxy),
# appended to ~/avesnova/runs-gpubox2/reach.log: the outside view of downtime, next to the
# box's own heartbeat (which cannot log while the box is stopped).
#   systemd-run --user --unit onehead-reach -p MemoryMax=256M reach.sh
LOG=$HOME/avesnova/runs-gpubox2/reach.log
while true; do
  d=down; p=down
  timeout 10 bash -c '</dev/tcp/80.188.223.200/30148' 2>/dev/null && d=up
  timeout 10 bash -c '</dev/tcp/ssh9.vast.ai/22629' 2>/dev/null && p=up
  echo "$(date +%FT%T%z) direct $d proxy $p" >> "$LOG"
  sleep 60
done
