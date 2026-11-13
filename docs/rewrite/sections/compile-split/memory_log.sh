#!/bin/bash
# memory_log.sh: every 30 s, each bench process's peak and current resident memory (VmHWM,
# VmRSS, kB) with its command's variant, and the cgroup's memory.current and memory.events,
# appended to /workspace/cs/memory.log.
while true; do
  line="$(TZ=Asia/Seoul date +%H:%M:%S) cgroup current $(($(cat /sys/fs/cgroup/memory.current) / 1048576)) MB, events $(tr '\n' ' ' < /sys/fs/cgroup/memory.events)"
  for pid in $(pgrep -f "[b]ench/compile_baseline"); do
    name=$(tr '\0' ' ' < /proc/$pid/cmdline | grep -o "config=[^ ]*\|set=[^ ]*" | tr '\n' ' ')
    line="$line | $pid $name HWM $(awk '/VmHWM/ {print $2}' /proc/$pid/status) RSS $(awk '/VmRSS/ {print $2}' /proc/$pid/status)"
  done
  echo "$line" >> /workspace/cs/memory.log
  sleep 30
done
