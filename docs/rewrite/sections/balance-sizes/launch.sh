#!/bin/bash
# launch.sh UNIT LOG ARGS...: sizes.py as a systemd user unit (MemoryMax=8G), logging to LOG
unit=$1; logfile=$2; shift 2
cd /home/vizia/avesnova/bnb-balance
systemd-run --user --unit="$unit" -p MemoryMax=8G --collect \
  --setenv=PYTHONPATH=src:. --setenv=VIRTUAL_ENV=/home/vizia/avesnova/boost-and-broadside-v2/.venv \
  --working-directory=/home/vizia/avesnova/bnb-balance \
  /bin/bash -c "exec /home/vizia/.local/bin/uv run --no-sync --active python $* > $logfile 2>&1"
