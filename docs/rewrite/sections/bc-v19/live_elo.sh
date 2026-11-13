#!/bin/bash
# Live Elo on a second GPU (the laptop) while a run trains elsewhere: every few minutes copy
# the run's newest policy snapshots (`policies/step_*/exponential_average`) from the training
# box, and rate each new one against the scripted agent (Baekho, 1000) on the three modes,
# reusing every earlier round's games (`bnb ratings --reuse`), so the curve accumulates and
# only the new snapshot plays. Pauses while `bnb play` runs (the play shell needs the GPU)
# or while $PAUSE exists.
#     REMOTE=root@host PORT=22 RUN=bc-v19 bash live_elo.sh
set -uo pipefail
: "${REMOTE:?}" "${PORT:?}" "${RUN:?}"
REMOTE_RUNS=${REMOTE_RUNS:-/workspace/runs}
HERE=${HERE:-$HOME/avesnova/live-elo/$RUN}
PAUSE=${PAUSE:-$HOME/avesnova/live-elo/pause}
SE=${SE:-35}
PRESETS="real_scale_elimination real_scale_deathmatch real_scale_frontline"
mkdir -p "$HERE/runs/$RUN/policies" "$HERE/rounds" "$HERE/cache"
export TORCHINDUCTOR_CACHE_DIR=$HERE/cache/inductor TRITON_CACHE_DIR=$HERE/cache/triton
export TRITON_DEFAULT_FP_FUSION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  if [ -e "$PAUSE" ] || pgrep -f "bin/bnb play" > /dev/null; then sleep 60; continue; fi
  rsync -a -e "ssh -p $PORT -o ConnectTimeout=20" "$REMOTE:$REMOTE_RUNS/$RUN/run.json" \
    "$REMOTE:$REMOTE_RUNS/$RUN/config.json" "$HERE/runs/$RUN/" 2> /dev/null
  rsync -a -e "ssh -p $PORT -o ConnectTimeout=20" --include='step_*/' \
    --include='step_*/exponential_average/***' --exclude='*' \
    "$REMOTE:$REMOTE_RUNS/$RUN/policies/" "$HERE/runs/$RUN/policies/" 2> /dev/null
  newest=$(ls -d "$HERE"/runs/"$RUN"/policies/step_*/exponential_average 2> /dev/null | sort -t_ -k2 -n | tail -1)
  if [ -z "$newest" ] || [ -e "$newest/.rated" ]; then sleep 300; continue; fi
  players=$(ls -d "$HERE"/runs/"$RUN"/policies/step_*/exponential_average | while read p; do [ -e "$p/.rated" ] && echo "$p"; done)
  step=$(basename "$(dirname "$newest")")
  echo "$(date +%H:%M) rating $step" >> "$HERE/live_elo.log"
  bnb ratings "$RUN" --runs-directory "$HERE/runs" --no-league --no-ladder \
    --with $players "$newest" --presets $PRESETS --target-stderr "$SE" --budget 768 \
    --blocks 2 --block-environments 96 --seed "${step#step_}" \
    --lineup Yongma Yongma Jujak Jujak Jujak \
    --reuse "$HERE/rounds" --out "$HERE/rounds/$step" >> "$HERE/live_elo.log" 2>&1 \
    && touch "$newest/.rated"
  # the last finished cycle's whole table (every mode, every snapshot)
  awk '/ rating step_/ { if (block != "") last = block; block = "" } /^real_scale_[a-z]+ / { block = block $0 "\n" } END { printf "%s", (block != "" ? block : last) }' \
    "$HERE/live_elo.log" > "$HERE/latest.txt"
  du -sh "$HERE/cache" >> "$HERE/live_elo.log"
done
