#!/bin/bash
# Supervisor program: run hf_backup.py every 30 minutes on the box (hf_backup.supervisor.conf).
# Without a token it waits quietly; the first backup follows within 30 minutes of the token
# arriving in /workspace/secrets.env (HF_TOKEN=...). Sourcing env.sh reads that file.
# shellcheck disable=SC1091
source /workspace/env.sh
HERE=$(cd "$(dirname "$0")" && pwd)
while true; do
  if [ -n "${HF_TOKEN:-}" ]; then
    python "$HERE/hf_backup.py" --profile box --lock /workspace/hf-backup/lock >> /workspace/logs/hf_backup.log 2>&1
    echo "$(date -u +%FT%TZ) hf_backup exit $?" >> /workspace/logs/hf_backup.log
  fi
  sleep 1800
  source /workspace/env.sh  # pick up a token written meanwhile
done
