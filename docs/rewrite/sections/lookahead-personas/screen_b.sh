#!/bin/bash
# Screen b: head-to-head of the three crowd measures' candidates and plain (the old yardstick
# saturates at 8v8 and 16v16), SE 0.02, same store as screen.sh.
D=docs/rewrite/sections/lookahead-personas
DATA=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2
C=$D/candidates
NAMED="plain=$C/plain.json crowd-own=$C/crowd-own.json crowd-known=$C/crowd-known.json crowd-near=$C/crowd-near.json"
for i in $(seq 1 10); do
  grep -q "verify done" $DATA/logs/screenb-*.log 2>/dev/null && break
  $D/locked2.sh screenb-$i "uv run --no-sync python $D/tune.py $DATA/screen verify $NAMED --opponents plain crowd-own crowd-known crowd-near --sizes 1 2 4 8 16 --target-error 0.02 --minutes 11"
done
