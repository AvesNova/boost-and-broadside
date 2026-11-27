#!/bin/bash
# Side jobs: 2v2 and 8v8 durations (study clocks for the step 4 checks), crowd measures.
D=docs/rewrite/sections/lookahead-personas
DATA=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2
S="uv run --no-sync python $D/clock.py $DATA/clock.sqlite play"
$D/locked2.sh clock2 "$S --size 2 --games 512 --step 512 --envs 4096 --minutes 10"
$D/locked2.sh clock8 "$S --size 8 --games 384 --step 384 --envs 2048 --minutes 10"
$D/locked2.sh crowd "uv run --no-sync python $D/crowd.py $DATA/crowd.json --sizes 1 2 4 8 16 --ticks 2700 --envs 256"
