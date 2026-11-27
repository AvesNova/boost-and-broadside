#!/bin/bash
# Round 3: play a group in locked jobs of about 10 minutes, at most N jobs (resumable).
#   round3_loop.sh GROUP N [play options...]
D=docs/rewrite/sections/lookahead-personas
DATA=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-3
group=$1; n=$2; shift 2
for i in $(seq 1 $n); do
  grep -q "$group done" $DATA/logs/$group-*.log 2>/dev/null && break
  $D/locked3.sh $group-$i "uv run --no-sync python $D/round3.py play $group --minutes 9 $*"
done
