#!/bin/bash
# The BC run on the newest game (2026-11-13 night, the 5090): game v19, Yemong v4, Baekho as
# the teacher, per-level BC returns, one map size per mode (arena_scale off), 2048
# environments (1.05M steps an update), 256 minibatches, 100M steps, no rating lanes,
# snapshots every 5M; micro-batches of 480 rows (a whole 48-view minibatch on the 5090). Start it from the repository root in tmux; logs to $RUNS/bc-v19.log.
set -euo pipefail
RUNS=${RUNS:-/workspace/runs}
CACHE=${CACHE:-/workspace/cache}
mkdir -p "$RUNS" "$CACHE/inductor" "$CACHE/triton" "$CACHE/bnb"
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton BNB_CACHE=$CACHE/bnb
export TORCHINDUCTOR_COMPILE_THREADS=8 TRITON_DEFAULT_FP_FUSION=1 PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "commit $(git rev-parse HEAD) $(date -u +%FT%TZ)" >> "$RUNS/bc-v19.log"
.venv/bin/bnb train --preset real_scale_all_modes --environments 2048 --seed 0 --name bc-v19 \
  --runs-directory "$RUNS" --update-policies False --objective bc \
  --set stop.budget_steps=100000000 --set lanes.environments=0 --set host_store=false \
  --set policy_snapshot_every=5000000 --set learner.micro_batch_rows=480 \
  --set 'lineup=["Yongma", "Yongma", "Jujak", "Jujak", "Jujak"]' --set arena_scale=null \
  >> "$RUNS/bc-v19.log" 2>&1
