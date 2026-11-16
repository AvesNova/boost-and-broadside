#!/bin/bash
# The balance check with the scripted agent's boundary fix (1d5cbcc3) applied; deathmatch without fog
# Frontline keep fog): every condition, 2048 games per entry of the given file(s), on the 5090
# box. Lines append to $OUT/<condition>.jsonl; entries already there are skipped.
#     bash docs/rewrite/sections/g-v23/run-fix.sh candidates.json [trims.json ...]
source /workspace/env.sh
cd /workspace/wt/g-v23
export PYTHONPATH=$PWD/src:$PWD   # this worktree, not env.sh's /workspace/bnb
OUT=/workspace/out/g-v23-fix; mkdir -p $OUT
for file in "$@"; do
  for c in deathmatch-5v5 deathmatch-10v10 elimination-5v5 elimination-10v10 frontline-5v5 frontline-10v10 deathmatch-2v2 deathmatch-1v1; do
    echo "$(date +%H:%M:%S) start $c $file"
    flock $GPU_LOCK python -u -m bench.balance_sweep sweep --condition $c \
      --candidates docs/rewrite/sections/g-v23/$file --out $OUT/$c.jsonl \
      --games 2048 --variants 2 --environments 4096 2>&1 | grep -v -i warn
    echo "$(date +%H:%M:%S) done $c $file"
  done
done
echo "=== all done"
