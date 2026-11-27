#!/bin/bash
# Step 2: locked skill jobs (each bounded to about 12 minutes) until every size reaches the
# target error (1v1 and 4v4 0.015, 16v16 0.015), or 12 jobs per size.
D=docs/rewrite/sections/lookahead-personas
DATA=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2
S="uv run --no-sync python $D/skill2.py $DATA/skill2.sqlite play --target-error 0.015 --minutes 11"
for size in 1 4 16; do
  for i in $(seq 1 12); do
    grep -q "^${size}v${size}: done" $DATA/logs/skill2-$size-*.log 2>/dev/null && break
    $D/locked2.sh skill2-$size-$i "$S --size $size --envs 4096"
  done
done
