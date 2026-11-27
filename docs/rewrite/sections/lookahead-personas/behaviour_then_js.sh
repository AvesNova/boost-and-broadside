#!/bin/bash
# The behaviour jobs (behaviour.sh), then the league candidates' JS on the same state sets.
D=docs/rewrite/sections/lookahead-personas
DATA=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2
$D/behaviour.sh
$D/locked2.sh league-js "uv run --no-sync python $D/league_js.py $DATA/behaviour-centre $D/league-js-2.json robust-centre yardstick crowd-known cma-top1 league1 league2 league3 league4 league5 league6"
