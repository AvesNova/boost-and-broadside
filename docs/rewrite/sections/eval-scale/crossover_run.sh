#!/usr/bin/env bash
# The crossover end to end with a network over several sizes, run twice on one store: the
# second run must play nothing (every match read back from the store) and print the same
# walk. Then Elo and styles at one unequal size read the crossover's games.
#   bash docs/rewrite/sections/eval-scale/crossover_run.sh OUT_DIR [CHECKPOINT]
# Without CHECKPOINT: an untrained network (initial weights, seed 0, the core arena's config)
# saved as OUT_DIR/untrained.pt.
set -euo pipefail
out=$1
checkpoint=${2:-$out/untrained.pt}
mkdir -p "$out"
if [ ! -f "$checkpoint" ]; then
  uv run --no-sync python -c "
import sys, torch
from lean import game, model
from lean.learn import LearnConfig, WeightAverages, save_checkpoint
torch.manual_seed(0)
policy = model.Policy(model.ModelConfig.for_arena(game.GameConfig().arena_radius))
save_checkpoint(sys.argv[1], policy, WeightAverages(policy, LearnConfig()), LearnConfig())
" "$checkpoint"
fi
store=$out/crossover.sqlite
games() { uv run --no-sync python -c "import sqlite3,sys; print(sqlite3.connect(sys.argv[1]).execute('SELECT COUNT(*) FROM games').fetchone()[0])" "$store"; }
rm -f "$store"*
powerprofilesctl get
for run in first second; do
  echo "== $run run $(date +%H:%M:%S)"
  start=$(date +%s.%N)
  uv run --no-sync python -m lean.eval crossover "$store" "$checkpoint" --weights live \
    --sizes 4 8 16 32 --games 64 2> "$out/crossover-$run.err"
  echo "seconds $(echo "$(date +%s.%N) - $start" | bc) games stored $(games)"
done
echo "== elo at 8 against 6 (network on 8)"
uv run --no-sync python -m lean.eval elo "$store" --sizes 8 6
echo "== the same, sizes turned round"
uv run --no-sync python -m lean.eval elo "$store" --sizes 6 8
echo "== styles, network on 8 against 6"
uv run --no-sync python -m lean.eval styles "$store" --sizes 8 6
echo "== report (yardstick first, sizes 6 8: the network's games read the other way round)"
uv run --no-sync python -m lean.eval report "$store" yardstick "net:$checkpoint:live" --sizes 6 8
