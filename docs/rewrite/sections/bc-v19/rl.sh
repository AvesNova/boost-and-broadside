#!/bin/bash
# RL from a BC run's averaged policy, on the 5090, with the champion's PPO settings: a warm
# start (weights and return scale, fresh optimizer), 2048 environments (about 1.05M steps an
# update), 32 minibatches, 4 epochs, per-level gamma/lambda defaults, 256 rating lanes against
# the league, snapshots every 5M (live Elo on the laptop), one map size per mode (as BC).
# Start it from the repository root in tmux named $NAME; logs to $RUNS/$NAME.log.
set -euo pipefail
NAME=${NAME:-rl-v20}
FROM=${FROM:-bc-v20}
UPDATES=${UPDATES:-143}  # about 150M steps
RUNS=${RUNS:-/workspace/runs}
CACHE=${CACHE:-/workspace/cache}
mkdir -p "$RUNS" "$CACHE/inductor" "$CACHE/triton" "$CACHE/bnb"
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton BNB_CACHE=$CACHE/bnb
export TORCHINDUCTOR_COMPILE_THREADS=8 TRITON_DEFAULT_FP_FUSION=1 PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "commit $(git rev-parse HEAD) $(date -u +%FT%TZ) from $FROM" >> "$RUNS/$NAME.log"
.venv/bin/bnb train --preset real_scale_all_modes --environments 2048 --seed 0 --name "$NAME" \
  --runs-directory "$RUNS" --init-from "$RUNS/$FROM/policy" --updates "$UPDATES" \
  --update-policies False --set lanes.environments=256 --set host_store=false \
  --set policy_snapshot_every=5000000 --set learner.micro_batch_rows=480 \
  --set 'lineup=["Yongma", "Yongma", "Jujak", "Jujak", "Jujak"]' --set arena_scale=null \
  >> "$RUNS/$NAME.log" 2>&1
