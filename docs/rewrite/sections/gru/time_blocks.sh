#!/bin/bash
# One arm's timing: 4 short PPO updates on the laptop; argument: the temporal block
# ("griffin" or "gru"), same settings for both. Logs in runs/gru-timing (ignored).
block=$1
cd /home/vizia/avesnova/bnb-gru
export PYTHONPATH=/home/vizia/avesnova/bnb-gru/src
S=/home/vizia/avesnova/bnb-gru/runs/gru-timing
mkdir -p $S
echo "=== $block $(date +%T) $(powerprofilesctl get)" >> $S/time_blocks.log
.venv/bin/python -c "import bnb.cli; bnb.cli.main()" train --preset real_scale_all_modes --environments 256 --updates 4 \
  --micro-batch-rows 96 --name time-$block --runs-directory $S/runs \
  --set lanes.environments=32 --set host_store=true --set arena_scale=null \
  --set yemong.temporal_block=$block > $S/time-$block.log 2>&1
status=$?
echo "=== $block exit $status $(date +%T)" >> $S/time_blocks.log
exit $status
