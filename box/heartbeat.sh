#!/bin/bash
# Supervisor program heartbeat: a UTC timestamp a minute to /workspace/heartbeat.log, so a gap
# is downtime (the box stopped or was outbid). Each start is marked with "boot".
echo "$(date -u +%FT%TZ) boot $(cat /proc/sys/kernel/random/boot_id) uptime $(cut -d' ' -f1 /proc/uptime)" >> /workspace/heartbeat.log
# each line: time, GPU utilisation %, memory used MiB, power W (nvidia-smi)
while true; do
  echo "$(date -u +%FT%TZ) $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" >> /workspace/heartbeat.log
  sleep 60
done
