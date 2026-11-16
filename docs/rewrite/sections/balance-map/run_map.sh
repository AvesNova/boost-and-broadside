#!/bin/bash
# The balance-map run on box5090: the search (bench.balance_map) and one persistent balance
# worker per searched condition (bench.balance_worker), each worker compiled in turn (one
# cold compile at a time: about 12 GB of RAM each) and sharing the GPU lock. A rerun resumes
# (the search from OUT/scheduler.pkl; workers skip finished entries).
#
#   ssh -f box5090 'setsid nohup bash /workspace/wt/map/docs/rewrite/sections/balance-map/run_map.sh \
#       OUT ARGS... >/dev/null 2>&1 </dev/null &'
#
# OUT: the run's directory; ARGS: `bench.balance_map search`'s (--generations, --hours, ...).
# CONDITIONS (environment): the workers' conditions (default the searched four).
set -uo pipefail
source /workspace/env.sh
WORKTREE=/workspace/wt/map
OUT=$1
shift
CONDITIONS=${CONDITIONS:-deathmatch-2v2 deathmatch-5v5 elimination-5v5 frontline-10v10}
export PYTHONPATH=$WORKTREE/src:$WORKTREE
cd "$WORKTREE" || exit 1
mkdir -p "$OUT/logs"
python -c "import bnb; print('bnb', bnb.__file__)" >> "$OUT/logs/run.log"
git -C "$WORKTREE" log --oneline -1 >> "$OUT/logs/run.log"
if [ "${1:-}" != "--workers-only" ]; then
  python -u -m bench.balance_map search --out "$OUT" "$@" >> "$OUT/logs/search.log" 2>&1 &
  search=$!
  echo "search $search" >> "$OUT/logs/pids"
fi
workers=()
for condition in $CONDITIONS; do
  log="$OUT/logs/worker-$condition.log"
  python -u -m bench.balance_worker --condition "$condition" --queue "$OUT/queue/$condition" \
    --lock "$GPU_LOCK" --environments "${ENVIRONMENTS:-4096}" --variants "${VARIANTS:-32}" \
    --prewarm-processes 4 >> "$log" 2>&1 &
  pid=$!
  workers+=("$pid")
  echo "worker-$condition $pid" >> "$OUT/logs/pids"
  echo "$(date +%H:%M:%S) worker $condition pid $pid compiling" >> "$OUT/logs/run.log"
  until grep -q "ready in" "$log"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$(date +%H:%M:%S) worker $condition died" >> "$OUT/logs/run.log"
      break
    fi
    sleep 5
  done
  echo "$(date +%H:%M:%S) worker $condition ready" >> "$OUT/logs/run.log"
done
if [ -n "${search:-}" ]; then
  wait "$search"
  echo "$(date +%H:%M:%S) search ended; stopping the workers" >> "$OUT/logs/run.log"
  kill "${workers[@]}" 2>/dev/null
fi
wait
