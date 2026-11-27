#!/bin/bash
# Step 4: CMA-ES in locked jobs of about 12 minutes; bound: 60 generations or 05:30.
# Rated against candidates/cma-reference.json: crowd-known (the screen's best mean; the old
# yardstick saturates) for generations 1-14, then the search mean after generation 16 (candidates/cma-mean-g16.json).
D=docs/rewrite/sections/lookahead-personas
DATA=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2
C=$D/candidates
for i in $(seq 1 40); do
  [ "$(date +%H%M)" -ge 0530 ] && [ "$(date +%H%M)" -lt 1200 ] && break
  [ -f $DATA/cma/cma.jsonl ] && [ $(wc -l < $DATA/cma/cma.jsonl) -ge 60 ] && break
  $D/locked2.sh cma-$i "uv run --no-sync python $D/tune.py $DATA/cma cma reference=$C/cma-reference.json --space $C/cma-space.json --minutes 10 --games 192 --popsize 12 --sigma 0.15"
done
