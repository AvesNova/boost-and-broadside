#!/bin/bash
# The deathmatch clock (2026-11-16): Yongma team v Jujak team at 2v2, 5v5 and 10v10 for the
# cap, 600 s, 4096 games each (seed 0, fog on, fields off); the kills by minute give the
# kill share had the clock stopped after each minute, from the same games. Waits for the
# timings (one compile at a time); the lock is held for the play only (the worker).
source /workspace/env.sh
here=/workspace/wt/balance-eval-clock
out=/workspace/out/balance-eval
export PYTHONPATH=$here/src:$here
cd $here
until grep -q " done$" $out/timing_5090.log; do sleep 20; done
echo "[{\"name\": \"base: Yongma team v Jujak team\"}]" > $out/clock-candidates.json
for condition in deathmatch-5v5 deathmatch-10v10 deathmatch-2v2; do
  queue=$out/queue-clock-$condition; mkdir -p $queue
  echo "{\"candidates\": \"$out/clock-candidates.json\", \"out\": \"$out/clock-$condition.jsonl\", \"games\": 4096, \"seed\": 0}" > $queue/01-clock.json
  echo "$(date +%H:%M:%S) clock $condition"
  /workspace/venv/bin/python -m bench.balance_worker --condition $condition --queue $queue \
    --environments 4096 --variants 32 --seconds 600 --once --lock $GPU_LOCK > $out/clock-$condition.log 2>&1
  grep "kill share" $out/clock-$condition.log
done
echo "$(date +%H:%M:%S) done"
