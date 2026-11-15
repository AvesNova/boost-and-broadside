#!/bin/bash
# One arm's timing: 4 short PPO updates on the laptop; argument: the temporal block
# ("griffin" or "gru"), same settings for both; optional second argument: the environments
# (256 default; 128 fits the 8G MemoryMax, as the pinned host store cannot swap). Logs in
# runs/gru-timing (ignored).
block=$1
environments=${2:-256}
cd /home/vizia/avesnova/bnb-gru
export PYTHONPATH=/home/vizia/avesnova/bnb-gru/src
S=/home/vizia/avesnova/bnb-gru/runs/gru-timing
mkdir -p $S
echo "=== $block $environments $(date +%T) $(powerprofilesctl get)" >> $S/time_blocks.log
.venv/bin/python -c "import bnb.cli; bnb.cli.main()" train --preset real_scale_all_modes --environments $environments --updates 4 \
  --micro-batch-rows 96 --name time-$block-$environments --runs-directory $S/runs \
  --set lanes.environments=32 --set host_store=true --set arena_scale=null \
  --set yemong.temporal_block=$block > $S/time-$block-$environments.log 2>&1
status=$?
echo "=== $block $environments exit $status $(date +%T)" >> $S/time_blocks.log
exit $status
