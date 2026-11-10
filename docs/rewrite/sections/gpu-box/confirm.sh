#!/bin/bash
# The laptop agent's confirmation (2026-11-09 03:20 KST): C-fixed's full command, with its
# learner.discount and learner.trace_decay (missing from every box run so far: the outcome
# level's target is then the real game result, gamma = lambda = 1), to 10.5M on the box's code.
# Expect outcome Brier skill at 5-15 s of about +0.1 to +0.2 at updates 6-10.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
# wait for fastdecay to honour its STOP file
while pgrep -f "fastdecay-3090 --preset" > /dev/null; do sleep 10; done
NAME=2026-11-09-bc-absolute-cfixed-learner-check
echo "$(date -u +%H:%M) start $NAME" >> $LOG
.venv/bin/bnb train --name $NAME --preset mixed --objective bc --bc-budget-steps 10485760 \
  --micro-batch-rows 2560 --set host_store=false --set yemong.order=temporal_first \
  --set "learner.discount=[0.991,0.995,0.995,0.999,1.0]" \
  --set "learner.trace_decay=[0.99,0.99,0.99,0.99,1.0]" > runs/$NAME.out 2>&1
status=$?
brier=$(python3 -c "
import json
rows=[json.loads(l) for l in open('runs/$NAME/metrics.jsonl')]
v=[r['outcome/brier_skill/5_15_seconds'] for r in rows if 6<=r['counters/updates']<=10]
print(' '.join(f'{x:+.3f}' for x in v), '| mean', f'{sum(v)/len(v):+.3f}')" 2>&1)
echo "$(date -u +%H:%M) done $NAME (exit $status): brier 5-15 s at updates 6-10: $brier" >> $LOG
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
