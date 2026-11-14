#!/bin/bash
# Live Elo into the shared rating pool (bnb pool; bnb.league.rating_pool), on a second GPU while
# a run trains elsewhere: every few minutes copy the run's newest policy snapshots
# (`policies/step_*/exponential_average`) from the training box, register each new one in the
# pool, and rate it there on the three modes against the pool's players (Baekho, 1000, and
# every earlier snapshot of every run in the pool), every stored game reused. Several machines
# may run this at once against one pool on a shared volume: each writes its own files.
# Pauses while `bnb play` runs (the play shell needs the GPU) or while $PAUSE exists.
#     REMOTE=root@host PORT=22 RUN=rl-v21 POOL=/shared/rating-pool bash live_elo_pool.sh
set -uo pipefail
: "${REMOTE:?}" "${PORT:?}" "${RUN:?}"
REMOTE_RUNS=${REMOTE_RUNS:-/workspace/runs}
POOL=${POOL:-$HOME/avesnova/rating-pool}
HERE=${HERE:-$HOME/avesnova/live-elo/$RUN}
PAUSE=${PAUSE:-$HOME/avesnova/live-elo/pause}
SE=${SE:-45}
BUDGET=${BUDGET:-768}
PRESETS="real_scale_elimination real_scale_deathmatch real_scale_frontline"
mkdir -p "$HERE/runs/$RUN/policies" "$HERE/cache"
export TORCHINDUCTOR_CACHE_DIR=$HERE/cache/inductor TRITON_CACHE_DIR=$HERE/cache/triton
export TRITON_DEFAULT_FP_FUSION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# the pool's conditions, once (an existing pool with the same conditions is opened)
bnb pool create "$POOL" --lineup Yongma Yongma Jujak Jujak Jujak || exit 1
while true; do
  if [ -e "$PAUSE" ] || pgrep -f "bin/bnb play" > /dev/null; then sleep 60; continue; fi
  rsync -a -e "ssh -p $PORT -o ConnectTimeout=20" "$REMOTE:$REMOTE_RUNS/$RUN/run.json" \
    "$REMOTE:$REMOTE_RUNS/$RUN/config.json" "$HERE/runs/$RUN/" 2> /dev/null
  rsync -a -e "ssh -p $PORT -o ConnectTimeout=20" --include='step_*/' \
    --include='step_*/exponential_average/***' --exclude='*' \
    "$REMOTE:$REMOTE_RUNS/$RUN/policies/" "$HERE/runs/$RUN/policies/" 2> /dev/null
  newest=$(ls -d "$HERE"/runs/"$RUN"/policies/step_*/exponential_average 2> /dev/null | sort -t_ -k2 -n | tail -1)
  if [ -z "$newest" ] || [ -e "$newest/.rated" ]; then sleep 300; continue; fi
  step=$(basename "$(dirname "$newest")")
  echo "$(date +%H:%M) rating $step" >> "$HERE/live_elo.log"
  bnb pool add "$POOL" "$newest" >> "$HERE/live_elo.log" 2>&1 \
    && bnb pool rate "$POOL" --presets $PRESETS --target-stderr "$SE" --budget "$BUDGET" \
      --blocks 2 --block-environments 96 --seed "${step#step_}" \
      --runs-directory "$HERE/runs" >> "$HERE/live_elo.log" 2>&1 \
    && touch "$newest/.rated"
  # the pool's whole table (every mode, every run's snapshots) and this run's curves
  bnb pool table "$POOL" --presets $PRESETS > "$HERE/latest.txt" 2>> "$HERE/live_elo.log"
  bnb pool curves "$POOL" --presets $PRESETS --out "$HERE/curves.csv" >> "$HERE/live_elo.log" 2>&1
  du -sh "$HERE/cache" >> "$HERE/live_elo.log"
done
