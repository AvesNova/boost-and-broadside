#!/bin/bash
# Step 1 jobs: durations per size (each locked job bounded to about 12 minutes).
D=docs/rewrite/sections/lookahead-personas
S="uv run --no-sync python $D/clock.py /home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2/clock.sqlite play"
$D/locked2.sh clock1 "$S --size 1 --games 1024 --step 512 --envs 4096 --minutes 10"
$D/locked2.sh clock4 "$S --size 4 --games 512 --step 256 --envs 2048 --minutes 10"
$D/locked2.sh clock16 "$S --size 16 --games 384 --step 128 --envs 1024 --minutes 10"
