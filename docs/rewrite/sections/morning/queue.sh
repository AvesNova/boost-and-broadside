#!/bin/bash
# queue.sh: the morning's GPU jobs one at a time as systemd user units (MemoryMax=8G), each
# after the previous unit ends and while no `bnb play` process runs (checked every 2 min).
W=/home/vizia/avesnova/bnb-morning
S=docs/rewrite/sections
LOG=$W/$S/morning/queue.log
PY="/home/vizia/.local/bin/uv run --no-sync --active python"
wait_unit() { while systemctl --user is-active --quiet "$1"; do sleep 15; done; }
wait_play() { while pgrep -f "bnb play" > /dev/null; do
  echo "$(date +%H:%M) play running, paused" >> $LOG; sleep 120; done; }
launch() { unit=$1; out=$2; shift 2
  wait_play
  echo "$(date +%H:%M) start $unit" >> $LOG
  systemd-run --user --unit="$unit" -p MemoryMax=8G --collect --quiet \
    --setenv=PYTHONPATH=src:. --setenv=VIRTUAL_ENV=/home/vizia/avesnova/boost-and-broadside-v2/.venv \
    --setenv=TRITON_DEFAULT_FP_FUSION=1 --working-directory=$W --property=StandardOutput=file:$out \
    --property=StandardError=file:$out "$@"
  sleep 5; wait_unit "$unit"; echo "$(date +%H:%M) end $unit" >> $LOG; }
launch morning-lead2 $W/$S/morning/lead/log.txt $PY $S/maneuvers-2/sizes.py \
  --candidates ../morning/lead/candidates.json --tag ../morning/lead/sizes --sizes 1 2 5 \
  --games 512 --lineups "energy v turn" --versus r9m2 --versus-sizes 1 2 5 \
  --versus-lineups "energy v turn" "turn v energy" "mixed" "mixed reversed"
echo "$(date +%H:%M) queue done" >> $LOG
