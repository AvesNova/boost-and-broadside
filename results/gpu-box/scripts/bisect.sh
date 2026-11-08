#!/bin/bash
# bisect.sh COMMIT: absolute (C-fixed's line) to 10.5M at COMMIT, from a detached worktree
# /workspace/bnb-at-COMMIT, with only the training-neutral speed settings. It reads the
# outcome head's Brier skill at 5-15 s over updates 8-10 (the laptop's C-fixed at 3b72a914:
# +0.18, +0.11, +0.19; current v2 on this box: -0.08 to -0.34).
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=$1; TREE=/workspace/bnb-at-$C
[ -d "$TREE" ] || git worktree add -q --detach "$TREE" "$C"
NAME=2026-11-09-bc-absolute-at-$C-check
echo "$(date -u +%H:%M) start $NAME (outcome Brier bisection)" >> runs/arms.log
PYTHONPATH=$TREE/src .venv/bin/python -c \
  "import sys; from bnb.cli import main; sys.argv = ['bnb'] + sys.argv[1:]; main()" \
  train --name $NAME --runs-directory /workspace/boost-and-broadside/runs \
  --preset mixed --objective bc --bc-budget-steps 10485760 --set yemong.order=temporal_first \
  --micro-batch-rows 2560 --set host_store=false > runs/$NAME.out 2>&1
status=$?
brier=$(python3 -c "
import json
rows=[json.loads(l) for l in open('runs/$NAME/metrics.jsonl')]
v=[r['outcome/brier_skill/5_15_seconds'] for r in rows if 8<=r['counters/updates']<=10]
print(' '.join(f'{x:+.3f}' for x in v), '| mean', f'{sum(v)/len(v):+.3f}')" 2>&1)
echo "$(date -u +%H:%M) done $NAME (exit $status): brier 5-15 s at updates 8-10: $brier" >> runs/arms.log
