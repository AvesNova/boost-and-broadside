#!/bin/bash
# Round 6, revised (the author, 09:50 KST): the two taus are coupled; nodecay and slowdecay wait
# for that design (one tau for the held rates, the hidden-row loss weight and the physics change
# applied to hidden rows). Meanwhile: hiddenphysics at seed 1 (the leader as it was trained).
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
NAME=2026-11-09-bc-deadreckon-split-hiddenphysics-seed1
say "start $NAME"
if .venv/bin/bnb train --name $NAME --seed 1 --preset mixed --objective bc \
    --bc-budget-steps 30000000 --micro-batch-rows 2560 --set host_store=false \
    --set yemong.order=temporal_first \
    --set "learner.discount=[0.991,0.995,0.995,0.999,1.0]" \
    --set "learner.trace_decay=[0.99,0.99,0.99,0.99,1.0]" \
    --set stop.score_target=1.0 --set stop.plateau_steps=1000000000 \
    --set yemong.next_state=delta --set yemong.spread_inputs=scalar --set learner.next_state=0.0033 \
    --set yemong.delta_baseline=extrapolation --set yemong.delta_split=true \
    --set learner.correction=0.0033 --set yemong.physics_hidden_decay=0.5 >> runs/$NAME.out 2>&1; then
  say "done $NAME"
else
  say "FAILED $NAME (runs/$NAME.out)"
fi
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
say "round6 seed1 done"
