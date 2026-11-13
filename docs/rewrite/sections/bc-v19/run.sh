#!/bin/bash
# The BC runs on the newest game (2026-11-13/14 nights, the 5090): game v19, Yemong at HEAD, Baekho as
# the teacher, per-level BC returns, one map size per mode (arena_scale off), 2048
# environments (1.05M steps an update), 256 minibatches, BUDGET steps, no rating lanes,
# snapshots every 5M; micro-batches of 480 rows (a whole 48-view minibatch on the 5090). Start it from the repository root in tmux named $NAME; logs to $RUNS/$NAME.log.
# bc-v19 (a87cbdb0, 100M) crashed at 27M on softmin(0, 0); bc-v20 is Yemong v8 with the fix, 150M.
set -euo pipefail
NAME=${NAME:-bc-v20}
BUDGET=${BUDGET:-150000000}
RUNS=${RUNS:-/workspace/runs}
CACHE=${CACHE:-/workspace/cache}
mkdir -p "$RUNS" "$CACHE/inductor" "$CACHE/triton" "$CACHE/bnb"
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton BNB_CACHE=$CACHE/bnb
export TORCHINDUCTOR_COMPILE_THREADS=8 TRITON_DEFAULT_FP_FUSION=1 PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "commit $(git rev-parse HEAD) $(date -u +%FT%TZ)" >> "$RUNS/$NAME.log"
.venv/bin/bnb train --preset real_scale_all_modes --environments 2048 --seed 0 --name "$NAME" \
  --runs-directory "$RUNS" --update-policies False --objective bc \
  --set stop.budget_steps="$BUDGET" --set lanes.environments=0 --set host_store=false \
  --set policy_snapshot_every=5000000 --set learner.micro_batch_rows=480 \
  --set 'lineup=["Yongma", "Yongma", "Jujak", "Jujak", "Jujak"]' --set arena_scale=null \
  >> "$RUNS/$NAME.log" 2>&1
