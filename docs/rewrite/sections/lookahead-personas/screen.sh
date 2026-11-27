#!/bin/bash
# Step 3/4 screen: the surrogate's candidates against the old yardstick at 1, 2, 4, 8, 16 per
# side (fresh store DATA/screen/verify.sqlite), target score SE 0.02; locked jobs of about 12
# minutes until done (at most 10).
D=docs/rewrite/sections/lookahead-personas
DATA=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2
C=$D/candidates
NAMED="plain=$C/plain.json plain-half=$C/plain-half.json crowd-own=$C/crowd-own.json crowd-known=$C/crowd-known.json crowd-near=$C/crowd-near.json crowd-own-half=$C/crowd-own-half.json"
for i in $(seq 1 10); do
  grep -q "verify done" $DATA/logs/screen-*.log 2>/dev/null && break
  $D/locked2.sh screen-$i "uv run --no-sync python $D/tune.py $DATA/screen verify $NAMED --sizes 1 2 4 8 16 --target-error 0.02 --minutes 11"
done
