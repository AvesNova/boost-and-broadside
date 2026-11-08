#!/bin/bash
# Round 3, part b: after part a (the Brier check, fastdecay): the outcome-Brier reproduction at
# C-fixed's commit 3b72a914 (the author's laptop: positive skill by 6-10M there; current v2
# here: negative), then deadreckon-split-fourier.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
until grep -q "round3a done" "$LOG"; do sleep 30; done

runs/bisect.sh 3b72a914

NAME=2026-11-09-bc-deadreckon-split-fourier-3090
say "start $NAME"
if .venv/bin/bnb train --name $NAME --preset mixed --objective bc --bc-budget-steps 30000000 \
    --micro-batch-rows 2560 --set host_store=false \
    --set yemong.order=temporal_first --set yemong.next_state=delta \
    --set yemong.delta_baseline=extrapolation --set yemong.spread_inputs=scalar \
    --set learner.next_state=0.0033 --set yemong.delta_split=true \
    --set learner.correction=0.0033 --set yemong.position_inputs=fourier \
    > runs/$NAME.out 2>&1; then
  say "done $NAME"
else
  say "FAILED $NAME (runs/$NAME.out)"
fi
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
say "round3b done"
