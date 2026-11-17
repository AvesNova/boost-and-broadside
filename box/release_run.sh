#!/bin/bash
# release_run.sh CHAIN: the release run on the 5090 box (box/setup.sh's layout), BC 100M then RL
# to 200M as one chain, under the GPU lock. Run detached from the pinned checkout:
#   ssh -f box5090 'nohup setsid bash /workspace/bnb/box/release_run.sh rel-jujak-s1 > /workspace/runs/rel-jujak-s1.launch.log 2>&1 < /dev/null &'
# Writes beside the chain's directory in $RUNS: CHAIN.log (the chain's output), CHAIN.pid (the
# trainer's own pid: the python process, `bnb train`), CHAIN.exit (its exit code), which
# box/train_watchdog.py reads. After a clean finish it exports the compile caches.
#
# The settings (the author, 2026-11-17; docs/rewrite/arms-todo.md): the preset
# release_jujak_elimination (`bnb.runtime.train.PRESET_SETTINGS`: real-scale elimination
# alone, clear, the hemisphere spawn, Jujak only, the belief off, the opponents' pools at
# 5v5 / 4v5 / 3v5 with the learner the smaller side, self-play and the rating lanes 5v5),
# and everything else as dry run 3 (box/dry_run.sh): 2048 environments, 480 micro-batch
# rows, segments on the device (host_store=false), every update's policy files in both
# stages (update_policies=all), BC without rating lanes, seed 0, W&B on. RL's 32 and BC's
# 256 minibatches are the stages' defaults. The box needs its own W&B login first.
set -uo pipefail
chain=${1:?usage: release_run.sh CHAIN}
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
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs 'machine api.wandb.ai' "$HOME/.netrc"; then
  echo "no W&B login on this box (WANDB_API_KEY or ~/.netrc): run /workspace/venv/bin/wandb login once" >&2
  exit 4
fi
cd "$BNB"
echo "commit $(git rev-parse HEAD) $(date -u +%FT%TZ)" > "$log"
bnb train --objective chain --chain-budget-steps 200000000 \
  --preset release_jujak_elimination --environments 2048 --seed 0 \
  --name "$chain" --runs-directory "$RUNS" --micro-batch-rows 480 \
  --set host_store=false --set update_policies=all --bc-set lanes.environments=0 --wandb \
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
