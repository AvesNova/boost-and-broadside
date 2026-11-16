#!/bin/bash
# Option G on v23 against v23 as is: every balance condition, 2048 games per candidate, on the
# 5090 box. One condition at a time (one compile each); lines append to $OUT/<condition>.jsonl.
source /workspace/env.sh
cd /workspace/wt/g-v23
OUT=/workspace/out/g-v23; mkdir -p $OUT
for c in deathmatch-5v5 elimination-5v5 frontline-5v5 deathmatch-2v2 deathmatch-1v1 deathmatch-10v10 elimination-10v10 frontline-10v10; do
  echo "$(date +%H:%M:%S) start $c"
  flock $GPU_LOCK python -u -m bench.balance_sweep sweep --condition $c \
    --candidates docs/rewrite/sections/g-v23/candidates.json --out $OUT/$c.jsonl \
    --games 2048 --variants 2 --environments 4096 2>&1 | grep -v -i warn
  echo "$(date +%H:%M:%S) done $c"
done
echo "=== all done"
