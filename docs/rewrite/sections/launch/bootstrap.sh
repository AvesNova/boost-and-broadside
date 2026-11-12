#!/bin/bash
# bootstrap.sh REF [CACHE_ARCHIVE]: set up a rented box (vast.ai, /workspace a volume) for
# the final experiments, on any number of GPUs. Run as root on the box:
#
#   curl -fsSL https://raw.githubusercontent.com/AvesNova/boost-and-broadside/REF/docs/rewrite/sections/launch/bootstrap.sh | bash -s REF
#   (or copy this file over and `bash bootstrap.sh REF [/workspace/compile-cache.tar.zst]`)
#
# REF: the branch or commit to train (the frozen game tree only: the compile cache is keyed
# by the code). Steps, each skipped when already done (the script runs again safely):
#   1. the repository: a clone of the public GitHub repository (no credentials), REF checked
#      out, at /workspace/boost-and-broadside
#   2. uv (if missing) and the locked environment (`uv sync --frozen`)
#   3. the caches on /workspace (env.sh: inductor, Triton, uv), compile threads per job
#   4. CACHE_ARCHIVE, if given: an earlier box's compile cache (cache.sh export), unpacked;
#      a file on the box, or hub:NAME (hub.sh pull-cache: from the Hugging Face hub)
#   5. the supervisor programs: the heartbeat and one queue runner per GPU (slot k, GPU k);
#      they start at boot, so the box resumes on its own after a restart; with a hub token
#      in /workspace/.env (HF_TOKEN=...), hub-sync too: the runs to the hub every 30 min
# It does not probe or queue: `launch.py probe` and `launch.py enqueue` next (runbook.md).
set -euo pipefail
REF=${1:?usage: bootstrap.sh REF [CACHE_ARCHIVE]}
ARCHIVE=${2:-}
URL=https://github.com/AvesNova/boost-and-broadside.git
REPO=/workspace/boost-and-broadside
HERE=docs/rewrite/sections/launch
say() { echo "$(date -u +%FT%TZ) bootstrap: $*" | tee -a /workspace/bootstrap.log; }
mkdir -p /workspace
say "start (ref $REF, host $(hostname), $(nproc) CPUs, $(nvidia-smi -L | wc -l) GPUs)"
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv | tee -a /workspace/bootstrap.log

# 1. the code
if [ ! -d $REPO/.git ]; then
  git clone --filter=blob:none --no-checkout $URL $REPO
fi
git -C $REPO fetch --quiet origin "$REF" || git -C $REPO fetch --quiet origin
git -C $REPO checkout --quiet --detach FETCH_HEAD 2>/dev/null || git -C $REPO checkout --quiet --detach "$REF"
say "code at $(git -C $REPO rev-parse --short HEAD) ($(git -C $REPO log -1 --format=%s | cut -c1-80))"

# 2. the environment
source $REPO/$HERE/env.sh
if ! command -v uv > /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/workspace/bin sh
fi
export PATH=/workspace/bin:$PATH
(cd $REPO && uv sync --frozen --quiet)
say "environment: $($REPO/.venv/bin/python -c 'import torch, triton; print("torch", torch.__version__, "triton", triton.__version__, "cuda", torch.version.cuda)')"

# 3-4. the caches
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" /workspace/runs /workspace/queue /workspace/logs
if [ "${ARCHIVE#hub:}" != "$ARCHIVE" ]; then
  bash $REPO/$HERE/hub.sh pull-cache "${ARCHIVE#hub:}" /workspace/hub
  ARCHIVE=$(ls /workspace/hub/*/caches/"${ARCHIVE#hub:}")
fi
if [ -n "$ARCHIVE" ]; then
  bash $REPO/$HERE/cache.sh import "$ARCHIVE"
fi

# 5. supervisor: the heartbeat and one runner per GPU
command -v supervisord > /dev/null || { apt-get update -qq && apt-get install -y -qq supervisor rsync zstd; }
command -v zstd > /dev/null || apt-get install -y -qq zstd
GPUS=$(nvidia-smi -L | wc -l)
CONF=/etc/supervisor/conf.d/bnb.conf
{
  echo "; written by bootstrap.sh: the heartbeat and one queue runner per GPU"
  echo "[program:heartbeat]"
  echo "command=$REPO/$HERE/heartbeat.sh"
  echo "autostart=true"
  echo "autorestart=true"
  echo "stdout_logfile=/workspace/logs/heartbeat.out"
  echo "redirect_stderr=true"
  for k in $(seq 0 $((GPUS - 1))); do
    echo
    echo "[program:runner-$k]"
    echo "command=$REPO/$HERE/runner.sh $k"
    echo "directory=$REPO"
    echo "autostart=true"
    echo "autorestart=true"
    echo "startsecs=10"
    # the runner stops its job gracefully (after the update in progress) and queues it again
    echo "stopsignal=TERM"
    echo "stopwaitsecs=900"
    echo "stopasgroup=false"
    echo "killasgroup=false"
    echo "stdout_logfile=/workspace/logs/runner-$k.out"
    echo "redirect_stderr=true"
  done
  if grep -q '^HF_TOKEN=' /workspace/.env 2>/dev/null; then
    echo
    echo "[program:hub-sync]"
    echo "command=bash -c 'source $REPO/$HERE/env.sh; exec $REPO/$HERE/hub.sh push-runs --loop 30'"
    echo "autostart=true"
    echo "autorestart=true"
    echo "stdout_logfile=/workspace/logs/hub-sync.out"
    echo "redirect_stderr=true"
  fi
} > $CONF
if pgrep -x supervisord > /dev/null; then
  supervisorctl reread > /dev/null && supervisorctl update
else
  supervisord -c /etc/supervisor/supervisord.conf
fi
sleep 2
supervisorctl status | tee -a /workspace/bootstrap.log
say "done: $GPUS runners on queue /workspace/queue/queue.json; next: launch.py check, probe, enqueue (runbook.md)"
