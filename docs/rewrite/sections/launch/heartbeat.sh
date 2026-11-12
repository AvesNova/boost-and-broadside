#!/bin/bash
# heartbeat.sh: a supervisor program. A UTC timestamp a minute to ${WORKSPACE:-/workspace}/heartbeat.log
# with each GPU's utilisation %, memory MiB and power W, so a gap is downtime. Each start
# is marked "boot".
echo "$(date -u +%FT%TZ) boot $(cat /proc/sys/kernel/random/boot_id) uptime $(cut -d' ' -f1 /proc/uptime)" >> ${WORKSPACE:-/workspace}/heartbeat.log
while true; do
  echo "$(date -u +%FT%TZ) $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr '\n' ' ')" >> ${WORKSPACE:-/workspace}/heartbeat.log
  sleep 60
done
