#!/bin/bash
# the baseline's trajectory images (frozen card, r9)
cd /home/vizia/avesnova/bnb-balance
T=docs/rewrite/sections/balance-sizes/trajectories.py
C=${1:-baseline/candidates.json}; P=${2:-frozen}; I=${3:-0}
for lineup in "energy v turn" "turn mirror" "energy mirror"; do
  short=$(echo $lineup | sed 's/energy v turn/evt/; s/turn mirror/jj/; s/energy mirror/yy/')
  uv run --no-sync --active python $T --candidates $C --candidate $I --size 1 --lineup "$lineup" --games 48 --plots 4 --out images/$P-1v1-$short
done
uv run --no-sync --active python $T --candidates $C --candidate $I --size 2 --lineup "energy v turn" --games 8 --plots 2 --out images/$P-2v2-evt
uv run --no-sync --active python $T --candidates $C --candidate $I --size 5 --lineup "energy v turn" --games 4 --plots 2 --window 60 --out images/$P-5v5-evt
