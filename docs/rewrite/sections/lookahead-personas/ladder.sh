#!/bin/bash
# Validation ladder: 16 agents, every pair, adaptive to score SE 0.05, 1v1/4v4/16v16 (fresh
# store DATA/ladder/ladder.sqlite); locked jobs of about 10 minutes, at most 8.
D=docs/rewrite/sections/lookahead-personas
DATA=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2
for i in $(seq 1 8); do
  grep -q "ladder done" $DATA/logs/ladder-*.log 2>/dev/null && break
  $D/locked2.sh ladder-$i "uv run --no-sync python $D/active.py ladder  robust-centre=docs/rewrite/sections/lookahead-personas/candidates/robust-centre.json surface-peak=docs/rewrite/sections/lookahead-personas/candidates/surface-peak.json cma-top1=docs/rewrite/sections/lookahead-personas/candidates/cma-top1.json cma-top2=docs/rewrite/sections/lookahead-personas/candidates/cma-top2.json cma-mean-g16=docs/rewrite/sections/lookahead-personas/candidates/cma-mean-g16.json yardstick=docs/rewrite/sections/lookahead-personas/candidates/yardstick.json crowd-known=docs/rewrite/sections/lookahead-personas/candidates/crowd-known.json r1-persona14=docs/rewrite/sections/lookahead-personas/candidates/r1-persona14.json r1-persona27=docs/rewrite/sections/lookahead-personas/candidates/r1-persona27.json r1-persona00=docs/rewrite/sections/lookahead-personas/candidates/r1-persona00.json league1=docs/rewrite/sections/lookahead-personas/candidates/league1.json league2=docs/rewrite/sections/lookahead-personas/candidates/league2.json league3=docs/rewrite/sections/lookahead-personas/candidates/league3.json league4=docs/rewrite/sections/lookahead-personas/candidates/league4.json league5=docs/rewrite/sections/lookahead-personas/candidates/league5.json league6=docs/rewrite/sections/lookahead-personas/candidates/league6.json --target-error 0.05 --minutes 9"
done
