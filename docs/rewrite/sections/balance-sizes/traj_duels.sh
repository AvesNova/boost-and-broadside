#!/bin/bash
# a candidate's duel images: the mixed 1v1 and the Jujak mirror (traj_duels.sh CANDIDATES INDEX PREFIX)
cd /home/vizia/avesnova/bnb-balance
T=docs/rewrite/sections/balance-sizes/trajectories.py
for lineup in "energy v turn" "turn mirror"; do
  short=$(echo $lineup | sed 's/energy v turn/evt/; s/turn mirror/jj/')
  uv run --no-sync --active python $T --candidates $1 --candidate $2 --size 1 --lineup "$lineup" --games 32 --plots 3 --out images/$3-1v1-$short
done
