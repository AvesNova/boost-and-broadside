#!/bin/bash
# Behaviour around the robust centre: the state sets, Saltelli (base 32) JS from the centre,
# label blur at spreads 0.25 / 0.5 / 1 (inert rows held: boundary, seek, field); after the
# ladder. Locked jobs of about 9 minutes, at most 4 per stage.
D=docs/rewrite/sections/lookahead-personas
DATA=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2
until grep -q "ladder done" $DATA/logs/ladder-*.log 2>/dev/null; do sleep 30; done
B="uv run --no-sync python $D/blur2.py $DATA/behaviour-centre $D/candidates/robust-centre.json"
for stage in states saltelli blur; do
  for i in $(seq 1 4); do
    $D/locked2.sh bc-$stage-$i "$B $stage --minutes 8 --drop boundary_weight seek_weight field_weight --tag -box"
    grep -q "time bound" $DATA/logs/bc-$stage-$i.log || break
  done
done
