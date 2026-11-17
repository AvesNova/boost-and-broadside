#!/bin/bash
# Set up (or restore) a rented vast.ai box on its persistent volume. Idempotent: run it on a
# fresh box, after a recycle (the container filesystem is wiped, the volume is not), or again
# to update. Everything that should survive lives under $VOLUME (default /workspace):
#
#   $VOLUME/bnb               the repository (cloned from public GitHub; never pushed from here)
#   $VOLUME/venv              the locked virtual environment (uv sync --extra tools)
#   $VOLUME/python            uv's Python installs (3.14)
#   $VOLUME/cache/{uv,inductor,triton,hf,bnb}   package, compile, hub and bnb caches
#   $VOLUME/runs  $VOLUME/out  $VOLUME/queue  $VOLUME/logs   run records, sweep outputs, job queues, logs
#   $VOLUME/env.sh            the environment every shell and service sources
#
#   bash setup.sh [BRANCH]    (default feat/imugi-boundary; the box clones it if missing)
#
# Then: `source /workspace/env.sh; cd $BNB`. GPU jobs share one lock, `flock $GPU_LOCK ...`.
# Compile caches are keyed by GPU model, torch/Triton version, code and shapes: keep a box's
# checkout at one commit for a sweep (docs/rewrite/arms-todo.md, 2026-11-16).
set -euo pipefail
VOLUME=${VOLUME:-/workspace}
BRANCH=${1:-feat/imugi-boundary}
REPOSITORY=https://github.com/AvesNova/boost-and-broadside.git

mkdir -p "$VOLUME"/{venv,python,runs,out,queue,logs} "$VOLUME"/cache/{uv,inductor,triton,hf,bnb}

cat > "$VOLUME/env.sh" <<EOF
# written by box/setup.sh; sourced by ~/.bashrc and the supervisor programs
export VOLUME=$VOLUME
export BNB=$VOLUME/bnb
export RUNS=$VOLUME/runs
export CACHE=$VOLUME/cache
export GPU_LOCK=/tmp/gpu.lock
export UV_CACHE_DIR=$VOLUME/cache/uv
export UV_PYTHON_INSTALL_DIR=$VOLUME/python
export UV_PROJECT_ENVIRONMENT=$VOLUME/venv
export TORCHINDUCTOR_CACHE_DIR=$VOLUME/cache/inductor
export TRITON_CACHE_DIR=$VOLUME/cache/triton
export HF_HOME=$VOLUME/cache/hf
export BNB_CACHE=$VOLUME/cache/bnb
export PYTHONPATH=$VOLUME/bnb/src:$VOLUME/bnb
export PATH=$VOLUME/venv/bin:\$PATH
# the W&B key, copied from the laptop (never committed; delete after the runs)
[ -f $VOLUME/secrets.env ] && . $VOLUME/secrets.env || true
EOF
grep -q "$VOLUME/env.sh" ~/.bashrc 2>/dev/null || echo "source $VOLUME/env.sh" >> ~/.bashrc
# shellcheck disable=SC1091
source "$VOLUME/env.sh"

if [ ! -d "$BNB/.git" ]; then
  git clone --filter=blob:none --branch "$BRANCH" "$REPOSITORY" "$BNB"
else
  git -C "$BNB" fetch --quiet origin
  echo "repository present at $(git -C "$BNB" rev-parse --short HEAD) on $(git -C "$BNB" branch --show-current); not moved (pin a commit per sweep; update by hand)"
fi

cd "$BNB"
uv sync --extra tools --quiet
python -c "import bnb, torch; print('bnb', bnb.__file__); print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

# the heartbeat as a supervisor program (a line a minute; gaps are downtime)
if command -v supervisorctl >/dev/null; then
  cat > /etc/supervisor/conf.d/bnb-heartbeat.conf <<EOF
[program:bnb-heartbeat]
command=$BNB/box/heartbeat.sh
autostart=true
autorestart=true
stdout_logfile=$VOLUME/logs/heartbeat.supervisor.log
redirect_stderr=true
EOF
  supervisorctl reread >/dev/null && supervisorctl update >/dev/null
  supervisorctl status bnb-heartbeat || true
fi
df -h "$VOLUME" | tail -1
echo "=== setup done"
