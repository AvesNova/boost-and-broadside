#!/bin/bash
# killpat.sh REGEX: SIGTERM every process whose command line matches REGEX, except this
# script, its caller and their ancestors (a pattern typed in the caller's own command line
# would otherwise match the caller: it killed my shell three times on 2026-11-08).
pattern=$1
skip=" $$ "
p=$$
while [ "$p" -gt 1 ]; do
  p=$(ps -o ppid= -p "$p" | tr -d ' ')
  [ -n "$p" ] || break
  skip="$skip$p "
done
for pid in $(pgrep -f -- "$pattern"); do
  case "$skip" in *" $pid "*) continue ;; esac
  echo "killing $pid: $(ps -o args= -p "$pid" | cut -c1-100)"
  kill "$pid"
done
