#!/bin/bash
# 20v20 diagnostics on the frozen card: the 10v10 arena, no fog, 600 s games
cd /home/vizia/avesnova/bnb-balance
S=docs/rewrite/sections/balance-sizes/sizes.py
uv run --no-sync --active python $S --candidates diag20/candidates.json --tag diag20/arena --sizes 20 --games 1024 --arena-scale 0.7071
uv run --no-sync --active python $S --candidates diag20/candidates.json --tag diag20/nofog --sizes 20 --games 1024 --fog 0
uv run --no-sync --active python $S --candidates diag20/candidates.json --tag diag20/long --sizes 20 --games 1024 --seconds 600
