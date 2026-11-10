#!/bin/bash
# Round 13 (the author, 15:10 KST): the nearest model's RL to 180.4M (172 updates, as the Yemong
# RL runs), from its best BC 120M policy ($1: live or exponential_average, chosen by
# tournament 9). Every setting as the Yemong RL runs but the model and the memory-only
# micro-batch (3840 rows = one per 384-view minibatch; 1920 if that does not fit).
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
INIT=runs/2026-11-10-bc-nearest/policies/step_120586240/$1
NAME=2026-11-10-rl-nearest
until grep -q "tournament9 done\|tournament9 FAILED" "$LOG"; do sleep 20; done
for rows in 3840 1920; do
  say "start $NAME (RL to 180.4M from $1, micro-batch $rows)"
  if .venv/bin/bnb train --name $NAME --model nearest --init-from $INIT --preset mixed \
      --updates 172 --micro-batch-rows $rows --set host_store=false >> runs/$NAME.out 2>&1; then
    say "done $NAME"; break
  fi
  if grep -q "OutOfMemory\|out of memory" runs/$NAME.out && [ $rows = 3840 ]; then
    say "$NAME out of memory at 3840 rows: retrying at 1920"; rm -rf runs/$NAME; continue
  fi
  say "FAILED $NAME (runs/$NAME.out)"; break
done
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
say "round13 done"
