#!/bin/bash
# Redirected study: skill around crowd-known (SE 0.03), then the behaviour set (states,
# Saltelli base 32, blur), each locked job about 10 minutes, at most 6 jobs per stage.
D=docs/rewrite/sections/lookahead-personas
DATA=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2
B="uv run --no-sync python $D/blur2.py $DATA/behaviour $D/candidates/crowd-known.json"
for size in 1 4 16; do
  for i in $(seq 1 6); do
    grep -q "^${size}v${size}: done" $DATA/logs/around-$size-*.log 2>/dev/null && break
    $D/locked2.sh around-$size-$i "uv run --no-sync python $D/around.py $DATA/around.sqlite play --size $size --minutes 9"
  done
done
for stage in states saltelli blur; do
  for i in $(seq 1 6); do
    $D/locked2.sh $stage-$i "$B $stage --minutes 9 --drop boundary_weight seek_weight field_weight --tag -box"
    grep -q "time bound" $DATA/logs/$stage-$i.log || break
  done
done
