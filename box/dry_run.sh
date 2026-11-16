#!/bin/bash
# dry_run.sh CHAIN: the v2 dry run on the 5090 box (box/setup.sh's layout), BC 100M then RL
# 100M as one chain, under the GPU lock. Run detached from the pinned checkout:
#   ssh -f box5090 'nohup setsid bash /workspace/bnb/box/dry_run.sh dry-s0 > /workspace/runs/dry-s0.launch.log 2>&1 < /dev/null &'
# Writes beside the chain's directory in $RUNS: CHAIN.log (the chain's output), CHAIN.pid (the
# trainer's own pid: the python process, `bnb train`), CHAIN.exit (its exit code), which
# box/train_watchdog.py reads. After a clean finish it exports the compile caches for the
# arm boxes (docs/rewrite/sections/launch/cache.sh).
#
# The settings (2026-11-16): the preset's defaults (real_scale_all_modes, both ship types drawn
# per slot, policy snapshots at 1-20M then every 5M, league snapshots every 5M to 50M then
# doubling, BC's 100M budget), plus what the 5090 runs bc-v20/rl-v20 measured: segments on
# the device (host_store=false), 480 micro-batch rows; BC without rating lanes (the snapshots
# are rated afterwards, docs/rewrite/dry-run.md). No W&B (no key on the box).
set -uo pipefail
chain=${1:?usage: dry_run.sh CHAIN}
source /workspace/env.sh
log=$RUNS/$chain.log pidfile=$RUNS/$chain.pid exitfile=$RUNS/$chain.exit
if [ -e "$RUNS/$chain" ] || [ -e "$pidfile" ]; then
  echo "$RUNS/$chain or $pidfile exists: choose another name" >&2
  exit 2
fi
exec 9> "$GPU_LOCK"
if ! flock -n 9; then
  echo "the GPU lock $GPU_LOCK is held: not launching" >&2
  exit 3
fi
cd "$BNB"
echo "commit $(git rev-parse HEAD) $(date -u +%FT%TZ)" > "$log"
bnb train --objective chain --chain-budget-steps 200000000 \
  --preset real_scale_all_modes --environments 2048 --seed 0 \
  --name "$chain" --runs-directory "$RUNS" --micro-batch-rows 480 \
  --set host_store=false --bc-set lanes.environments=0 \
  >> "$log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$pidfile"
code=0
wait "$pid" || code=$?
echo "$code" > "$exitfile"
echo "exit $code $(date -u +%FT%TZ)"
if [ "$code" = 0 ]; then
  PYTHON=/workspace/venv/bin/python bash "$BNB/docs/rewrite/sections/launch/cache.sh" export \
    "/workspace/compile-cache-5090-$(git rev-parse --short=8 HEAD).tar.zst"
fi
