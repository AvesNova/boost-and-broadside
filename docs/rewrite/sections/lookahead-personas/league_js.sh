#!/bin/bash
# After the behaviour jobs: the league candidates' pairwise JS on the centre's state sets.
D=docs/rewrite/sections/lookahead-personas
DATA=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2
until ! pgrep -u vizia -x behaviour.sh >/dev/null && ! ps -u vizia -o args | grep -q "[l]ookahead-personas/behaviour.sh"; do sleep 30; done
$D/locked2.sh league-js "uv run --no-sync python $D/league_js.py $DATA/behaviour-centre $D/league-js-2.json robust-centre yardstick crowd-known cma-top1 league1 league2 league3 league4 league5 league6"
