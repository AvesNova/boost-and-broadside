#!/bin/bash
# Run jobs one after another, each its own unit (run.sh); a job line: UNIT SCRIPT ARGS...
# Waits for every bnb-* unit to finish before starting the next; logs the load and profile.
cd /home/vizia/avesnova/boost-and-broadside/v2-teacher6  # run.sh takes WORKTREE (default this one)
while read -r unit rest; do
  [ -z "$unit" ] && continue
  while systemctl --user list-units 'bnb-*' --state=active,activating --no-legend --plain | grep -q .; do sleep 5; done
  echo "$(date +%H:%M:%S) start $unit load $(cut -d' ' -f1 /proc/loadavg) $(powerprofilesctl get)"
  WORKTREE="${WORKTREE:-$PWD}" docs/rewrite/sections/teacher-6/run.sh $unit $rest >/dev/null
  sleep 2
  while systemctl --user is-active -q "bnb-6-$unit"; do sleep 5; done
  echo "$(date +%H:%M:%S) done $unit $(systemctl --user show -p Result --value bnb-6-$unit)"
done
