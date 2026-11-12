#!/bin/bash
# queue.sh: the morning's GPU jobs one at a time as systemd user units (MemoryMax=8G),
# each after the previous unit ends and while no `bnb play` process runs (checked every 2 min).
W=/home/vizia/avesnova/bnb-morning
LOG=$W/docs/rewrite/sections/morning/queue.log
wait_unit() { while systemctl --user is-active --quiet "$1"; do sleep 15; done; }
wait_play() { while pgrep -f "bnb play" > /dev/null; do
  echo "$(date +%H:%M) play running, paused" >> $LOG; sleep 120; done; }
launch() { unit=$1; out=$2; shift 2
  wait_play
  echo "$(date +%H:%M) start $unit" >> $LOG
  systemd-run --user --unit="$unit" -p MemoryMax=8G --collect \
    --setenv=PYTHONPATH=src:. --setenv=VIRTUAL_ENV=/home/vizia/avesnova/boost-and-broadside-v2/.venv \
    --setenv=TRITON_DEFAULT_FP_FUSION=1 --working-directory=$W \
    /bin/bash -c "exec /home/vizia/.local/bin/uv run --no-sync --active python $* > $out 2>&1"
  sleep 5; wait_unit "$unit"; echo "$(date +%H:%M) end $unit" >> $LOG; }
wait_unit morning-h2h-bc2
S=docs/rewrite/sections
launch morning-lead $W/$S/morning/lead/log.txt $S/maneuvers-2/sizes.py \
  --candidates ../morning/lead/candidates.json --tag ../morning/lead/sizes --sizes 1 2 5 \
  --games 512 --lineups "energy v turn" --versus r9 r9m2 --versus-sizes 1 2 5 \
  --versus-lineups "energy v turn" "turn v energy" "mixed" "mixed reversed"
launch morning-fourship $W/$S/morning/fourship.log $S/morning/fourship.py \
  --players r9 r9m2 --games 1024 --tag fourship
echo "$(date +%H:%M) queue done" >> $LOG
