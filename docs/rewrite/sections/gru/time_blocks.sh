#!/bin/bash
# griffin then gru: 4 short PPO updates each on the laptop, same settings
cd /home/vizia/avesnova/bnb-gru
export PYTHONPATH=/home/vizia/avesnova/bnb-gru/src
S=/tmp/claude-1000/-home-vizia-avesnova-boost-and-broadside-v2/09f3b7d2-4ece-4315-b115-64ea2461eb64/scratchpad
for block in griffin gru; do
  echo "=== $block $(date +%T)" >> $S/time_blocks.log
  .venv/bin/python -c "import bnb.cli, sys; print(bnb.cli.__file__, file=sys.stderr); bnb.cli.main()" train --preset real_scale_all_modes --environments 256 --updates 4 \
    --micro-batch-rows 96 --name time-$block --runs-directory $S/runs --update-policies False \
    --set lanes.environments=32 --set host_store=true --set arena_scale=null \
    --set yemong.temporal_block=$block > $S/time-$block.log 2>&1
  echo "=== $block exit $? $(date +%T)" >> $S/time_blocks.log
done
