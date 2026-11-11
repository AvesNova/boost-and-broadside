#!/bin/bash
# Laptop side: pull the box's runs (checkpoints, policies, records, logs), the heartbeat and
# the supervisor logs to ~/avesnova/runs-gpubox2/ every N minutes (default 30), trying the
# direct address, then the vast proxy (the direct port can change after an interruption).
# Each pass appends one line (reachability + the status) to pull.log. Run it as a user unit:
#   systemd-run --user --unit onehead-pull -p MemoryMax=2G pull.sh 30
set -u
DEST=$HOME/avesnova/runs-gpubox2
mkdir -p "$DEST/runs" "$DEST/box"
every=${1:-30}
while true; do
  host=
  for h in gpubox2 gpubox2-proxy; do
    if timeout 60 ssh -o ConnectTimeout=20 -o BatchMode=yes $h true 2>/dev/null; then host=$h; break; fi
  done
  stamp=$(date +%FT%T%z)
  if [ -z "$host" ]; then
    echo "$stamp unreachable (gpubox2 and gpubox2-proxy)" >> "$DEST/pull.log"
  else
    timeout 3000 rsync -a --partial -e "ssh -o ConnectTimeout=20" \
      $host:/workspace/boost-and-broadside/runs/ "$DEST/runs/" 2>> "$DEST/pull.err"
    r=$?
    timeout 300 rsync -a $host:/workspace/heartbeat.log $host:/var/log/portal/onehead.log \
      "$DEST/box/" 2>> "$DEST/pull.err"
    status=$(timeout 60 ssh $host 'cd /workspace/boost-and-broadside && .venv/bin/python docs/rewrite/sections/gpubox2/status.py runs; supervisorctl status onehead heartbeat' 2>/dev/null | grep -v -E "Welcome|Have fun|AI agents" | tr '\n' '|')
    echo "$stamp via $host rsync=$r $status" >> "$DEST/pull.log"
  fi
  sleep $((every * 60))
done
