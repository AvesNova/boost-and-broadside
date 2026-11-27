#!/bin/bash
# Run jobs one after another, each its own unit (run.sh); a job line: UNIT SCRIPT ARGS...
# Waits for any bnb-* unit to finish before starting the next; logs a line per job.
cd /home/vizia/avesnova/boost-and-broadside/v2-review4
while read -r unit rest; do
  [ -z "$unit" ] && continue
  while systemctl --user list-units 'bnb-*' --state=active,activating --no-legend --plain | grep -q .; do sleep 5; done
  echo "$(date +%H:%M:%S) start $unit load $(cut -d' ' -f1 /proc/loadavg) $(powerprofilesctl get)"
  docs/rewrite/sections/training-review-2/run.sh $unit $rest >/dev/null
  sleep 2
  while systemctl --user is-active -q "bnb-4-$unit"; do sleep 5; done
  echo "$(date +%H:%M:%S) done $unit $(systemctl --user show -p Result --value bnb-4-$unit)"
done
