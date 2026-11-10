#!/bin/bash
# Round 12 (the author, 11:00 KST): the simple-model A/B. After the champion's RL reaches
# 180.4M: nearest BC to 120M, every training setting equal to the Yemong BC runs (only the model
# differs; the micro-batch is already one per minibatch). RL is NOT started here: the BC is
# checked first (first updates, ~10M, ~30M, a rating at 120M), the author's bug-hunt gates.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
until grep -q "round11 done" "$LOG"; do sleep 30; done
NAME=2026-11-10-bc-nearest
say "start $NAME (BC 120M, the simple-model A/B)"
if .venv/bin/bnb train --name $NAME --model nearest --preset mixed --objective bc \
    --bc-budget-steps 120000000 --set yemong.order=temporal_first \
    --set "learner.discount=[0.991,0.995,0.995,0.999,1.0]" \
    --set "learner.trace_decay=[0.99,0.99,0.99,0.99,1.0]" \
    --set stop.score_target=1.0 --set stop.plateau_steps=1000000000 \
    --micro-batch-rows 2560 --set host_store=false >> runs/$NAME.out 2>&1; then
  say "done $NAME"
else
  say "FAILED $NAME (runs/$NAME.out)"
fi
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
say "round12 done"
