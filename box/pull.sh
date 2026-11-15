#!/bin/bash
# Laptop side: pull a box's runs (policies, records, logs; not the training checkpoints,
# which only the box's resume needs and which the vast proxy moves at ~1 MB/s), the heartbeat
# and the supervisor log every N minutes, trying each host in turn; rsync --partial, retried,
# since the proxy closes long connections. Each pass appends one line (reachability + the
# status) to DEST/pull.log. Run it as a user unit:
#   systemd-run --user --unit onehead-pull3 -p MemoryMax=2G pull.sh 30 ~/avesnova/runs-gpubox3 gpubox3-direct gpubox3
set -u
every=${1:-30}; DEST=${2:-$HOME/avesnova/runs-gpubox2}; shift 2
HOSTS=("${@:-gpubox2 gpubox2-proxy}")
mkdir -p "$DEST/runs" "$DEST/box"
while true; do
  host=
  for h in ${HOSTS[@]}; do
    if timeout 60 ssh -n -o ConnectTimeout=20 -o BatchMode=yes $h true 2>/dev/null; then host=$h; break; fi
  done
  stamp=$(date +%FT%T%z)
  if [ -z "$host" ]; then
    echo "$stamp unreachable (${HOSTS[*]})" >> "$DEST/pull.log"
  else
    for attempt in 1 2 3 4 5 6; do
      timeout 1800 rsync -a --partial --exclude "/*/checkpoints/" -e "ssh -o ConnectTimeout=20" \
        $host:/workspace/boost-and-broadside/runs/ "$DEST/runs/" 2>> "$DEST/pull.err" && break
    done
    r=$?
    timeout 300 rsync -a $host:/workspace/heartbeat.log $host:/var/log/portal/onehead.log \
      "$DEST/box/" 2>> "$DEST/pull.err"
    status=$(timeout 60 ssh -n $host 'cd /workspace/boost-and-broadside && .venv/bin/python docs/rewrite/sections/gpubox2/status.py runs; supervisorctl status onehead heartbeat' 2>/dev/null | grep -v -E "Welcome|Have fun|AI agents" | tr '\n' '|')
    echo "$stamp via $host rsync=$r (attempt $attempt) $status" >> "$DEST/pull.log"
  fi
  sleep $((every * 60))
done
