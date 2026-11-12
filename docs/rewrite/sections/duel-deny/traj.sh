#!/bin/bash
# the Jujak-mirror 1v1 on D2, before (r9) and after (duel deny, crowd 8): confirm candidates 6 and 7
cd /home/vizia/avesnova/bnb-duel-deny
T=docs/rewrite/sections/duel-deny/trajectories.py
uv run --no-sync --active python $T --candidates confirm/candidates.json --candidate 6 --size 1 --lineup "turn mirror" --games 32 --plots 2 --out images/D2-jj-r9
uv run --no-sync --active python $T --candidates confirm/candidates.json --candidate 7 --size 1 --lineup "turn mirror" --games 32 --plots 2 --out images/D2-jj-duel-deny
