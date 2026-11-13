#!/bin/bash
# pull.sh DEST HOST PORT [EVERY_MINUTES]: laptop side. Pull a box's results every N minutes
# (once with EVERY 0) into DEST: the runs (records, metrics, logs, the tournament records
# and leaderboards, and the policy files the tournament field names; not the training
# checkpoints, which only the box's resumes need, nor the other policy snapshots), the
# queue and the job logs (beside it), the heartbeat and the bootstrap log. Each pass
# appends a line to DEST/pull.log with the box's `launch.py status`, and writes it to DEST/status.txt.
#
# HOST PORT: the box's DIRECT address (vast's "Direct ssh connect": root@IP -p PORT), not
# the ssh relay (sshN.vast.ai), which drops any transfer longer than about 2 minutes.
# HOST "local": DEST pulls from a directory on this machine, PORT then being that
# workspace (the laptop's end-to-end test).
#   systemd-run --user --unit bnb-pull -p MemoryMax=2G pull.sh ~/avesnova/runs-5090 root@1.2.3.4 41234 20
set -u
DEST=${1:?usage: pull.sh DEST HOST PORT [EVERY_MINUTES]}
HOST=${2:?}; PORT=${3:?}; EVERY=${4:-20}
mkdir -p "$DEST/runs" "$DEST/box"
if [ "$HOST" = local ]; then
  W=$PORT; SRC=""; SSH=(); RSH=()
  run() { bash -c "$1"; }
else
  W=/workspace; SRC="$HOST:"
  SSH=(ssh -n -p "$PORT" -o ConnectTimeout=20 -o BatchMode=yes -o ServerAliveInterval=30 "$HOST")
  RSH=(-e "ssh -p $PORT -o ConnectTimeout=20 -o BatchMode=yes -o ServerAliveInterval=30")
  run() { timeout 120 "${SSH[@]}" "$1"; }
fi
REPO=$W/boost-and-broadside
[ "$HOST" = local ] && REPO=${REPOSITORY:-$REPO}
while true; do
  stamp=$(date +%FT%T%z)
  if ! run true 2>/dev/null; then
    echo "$stamp unreachable ($HOST $PORT)" >> "$DEST/pull.log"
  else
    # the tournament's field: its policy files, which the pull keeps
    run "cat $W/runs/ratings/field.txt 2>/dev/null" | sed "s#^$W/runs/##" > "$DEST/field.txt"
    for attempt in 1 2 3 4 5; do
      timeout 3600 rsync -a --partial "${RSH[@]}" \
        --exclude "/*/checkpoints/" \
        --include-from=<(sed -E 's#^(.*)/([^/]+)$#/\1/\n/\1/\2/***#' "$DEST/field.txt") \
        --exclude "/*/policies/step_*/" \
        "$SRC$W/runs/" "$DEST/runs/" 2>> "$DEST/pull.err"
      r=$?
      [ $r = 0 ] && break
      sleep 30
    done
    timeout 600 rsync -a "${RSH[@]}" "$SRC$W/queue" "$SRC$W/heartbeat.log" "$SRC$W/bootstrap.log" \
      "$DEST/box/" 2>> "$DEST/pull.err"
    status=$(run "cd $REPO && WORKSPACE=$W ${PYTHON:-.venv/bin/python} docs/rewrite/sections/launch/launch.py status --runs $W/runs --queue $W/queue/queue.json" 2>&1)
    echo "$status" > "$DEST/status.txt"
    echo "$stamp rsync=$r (attempt $attempt) $(echo "$status" | grep -c running) running" >> "$DEST/pull.log"
  fi
  [ "$EVERY" = 0 ] && break
  sleep $((EVERY * 60))
done
