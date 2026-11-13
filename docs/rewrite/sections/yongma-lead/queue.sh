#!/bin/bash
# queue.sh JOB...: GPU jobs one at a time as systemd user units (MemoryMax=8G), each after
# the previous unit ends and while no `bnb play` process runs (`python -m bnb play` or the
# `.venv/bin/bnb play` entry point; checked every 2 min). The first version matched only
# `python -m bnb play` and missed the entry point (09:27 on 2026-11-13).
W=/home/vizia/avesnova/bnb-yongma-lead
S=docs/rewrite/sections
LOG=$W/$S/yongma-lead/queue.log
PY="/home/vizia/.local/bin/uv run --no-sync --active python"
wait_unit() { while systemctl --user is-active --quiet "$1"; do sleep 15; done; }
wait_play() { while pgrep -f "(^|/)(python[0-9.]*( .*)? -m bnb|bnb) play" > /dev/null; do
  echo "$(date +%H:%M) play running, paused" >> $LOG; sleep 120; done; }
launch() { unit=$1; out=$2; shift 2
  wait_play
  echo "$(date +%H:%M) start $unit (power $(powerprofilesctl get))" >> $LOG
  systemd-run --user --unit="$unit" -p MemoryMax=8G --collect --quiet \
    --setenv=PYTHONPATH=src:. --setenv=VIRTUAL_ENV=/home/vizia/avesnova/boost-and-broadside-v2/.venv \
    --setenv=TRITON_DEFAULT_FP_FUSION=1 --working-directory=$W --property=StandardOutput=file:$out \
    --property=StandardError=file:$out "$@"
  sleep 5; wait_unit "$unit"; echo "$(date +%H:%M) end $unit" >> $LOG; }
case $1 in
main) launch yongma-lead-main $W/$S/yongma-lead/log.txt $PY $S/maneuvers-2/sizes.py \
  --candidates ../yongma-lead/candidates.json --tag ../yongma-lead/sizes --sizes 1 2 5 10 \
  --games 512 --lineups "energy v turn" "energy mirror" --mirror-sizes 1 \
  --versus r9m2 --versus-sizes 1 5 \
  --versus-lineups "energy v turn" "turn v energy" "mixed" "mixed reversed" ;;
confirm) launch yongma-lead-confirm $W/$S/yongma-lead/confirm-log.txt $PY $S/maneuvers-2/sizes.py \
  --candidates ../yongma-lead/confirm.json --tag ../yongma-lead/confirm --sizes 1 2 5 10 \
  --games 1024 --lineups "energy v turn" "energy mirror" --mirror-sizes 1 \
  --versus r9m2 --versus-sizes 1 5 \
  --versus-lineups "energy v turn" "turn v energy" "mixed" "mixed reversed" ;;
last) wait_unit yongma-lead-confirm
  launch yongma-lead-last $W/$S/yongma-lead/last-log.txt $PY $S/maneuvers-2/sizes.py \
  --candidates ../yongma-lead/last.json --tag ../yongma-lead/last --sizes 1 2 5 10 \
  --games 512 --lineups "energy v turn" "energy mirror" --mirror-sizes 1 \
  --versus r9m2 --versus-sizes 1 5 10 \
  --versus-lineups "energy v turn" "turn v energy" "mixed" "mixed reversed" ;;
esac
echo "$(date +%H:%M) queue $1 done" >> $LOG
