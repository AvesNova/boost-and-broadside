#!/bin/bash
# Step 2: locked skill jobs (each bounded to about 12 minutes) until every size reaches the
# target error (1v1 and 4v4 0.015; 16v16 0.02 in rounds of 8,192: 16v16 plays about
# 22 games/s on the laptop), or 12 jobs per size.
D=docs/rewrite/sections/lookahead-personas
DATA=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2
S="uv run --no-sync python $D/skill2.py $DATA/skill2.sqlite play --minutes 9"
for size in 1 4 16; do
  for i in $(seq 1 12); do
    grep -q "^${size}v${size}: done" $DATA/logs/skill2-$size-*.log 2>/dev/null && break
    if [ $size -eq 16 ]; then extra="--target-error 0.02 --round-games 8192"; else extra="--target-error 0.015"; fi
    $D/locked2.sh skill2-$size-$i "$S --size $size --envs 4096 $extra"
  done
done
