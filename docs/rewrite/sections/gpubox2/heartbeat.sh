#!/bin/bash
# Supervisor program heartbeat: a UTC timestamp a minute to /workspace/heartbeat.log, so a gap
# is downtime (the box stopped or was outbid). Each start is marked with "boot".
echo "$(date -u +%FT%TZ) boot $(cat /proc/sys/kernel/random/boot_id) uptime $(cut -d' ' -f1 /proc/uptime)" >> /workspace/heartbeat.log
while true; do date -u +%FT%TZ >> /workspace/heartbeat.log; sleep 60; done
