#!/bin/bash
# Watch a training run: every 10 minutes copy its metrics from the training box and run the
# checklist (checklist.py) with the live Elo table; keep every report. Exit (so the caller is
# woken) when a FLAG appears that the previous report did not have, or when the run has made
# another WAKE_STEPS of progress since the last wake, or when the run's tmux session is gone.
#     REMOTE=root@host PORT=22 RUN=bc-v19 bash watch.sh
set -uo pipefail
: "${REMOTE:?}" "${PORT:?}" "${RUN:?}"
REMOTE_RUNS=${REMOTE_RUNS:-/workspace/runs}
HERE=${HERE:-$HOME/avesnova/live-elo/$RUN/watch}
ELO=${ELO:-$HOME/avesnova/live-elo/$RUN/latest.txt}
WAKE_STEPS=${WAKE_STEPS:-10000000}
CHECK=$(dirname "$0")/checklist.py
mkdir -p "$HERE"
woke=$(cat "$HERE/woke_steps" 2> /dev/null || echo 0)
while true; do
  rsync -a -e "ssh -p $PORT -o ConnectTimeout=20" "$REMOTE:$REMOTE_RUNS/$RUN/metrics.jsonl" "$HERE/" 2> /dev/null
  alive=$(ssh -p "$PORT" -o ConnectTimeout=20 "$REMOTE" "tmux has-session -t '=$RUN' 2>/dev/null && echo yes" 2> /dev/null | grep -c yes)
  report="$HERE/report-$(date +%m%d-%H%M).txt"
  python3 "$CHECK" "$HERE/metrics.jsonl" "$ELO" > "$report" 2>&1
  cp "$report" "$HERE/latest-report.txt"
  flags=$(grep '^FLAG' "$report" | cut -d: -f1 | sort)
  new=$(comm -13 <(sort "$HERE/flags" 2> /dev/null) <(echo "$flags") | grep -c FLAG)
  echo "$flags" > "$HERE/flags"
  steps=$(python3 -c "import json;r=[json.loads(l) for l in open('$HERE/metrics.jsonl')];print(int(r[-1].get('steps', len(r)*1048576)))" 2> /dev/null || echo 0)
  if [ "$new" -gt 0 ]; then echo "WAKE new flags: $report"; exit 0; fi
  if [ "$alive" = 0 ]; then echo "WAKE run session gone: $report"; exit 0; fi
  if [ $((steps - woke)) -ge "$WAKE_STEPS" ]; then echo "$steps" > "$HERE/woke_steps"; echo "WAKE progress $steps: $report"; exit 0; fi
  sleep 600
done
