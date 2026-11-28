#!/bin/bash
# Run jobs one after another, each its own unit (run.sh); a job line: UNIT WORKTREE SCRIPT ARGS...
# Waits for every bnb-* unit to finish before starting the next; logs the load and profile.
#   docs/rewrite/sections/fixes-10/queue.sh < jobs.txt
here=/home/vizia/avesnova/boost-and-broadside/v2-fixes10/docs/rewrite/sections/fixes-10
while read -r unit worktree rest; do
  [ -z "$unit" ] && continue
  while systemctl --user list-units 'bnb-*' --state=active,activating --no-legend --plain | grep -q .; do sleep 5; done
  echo "$(date +%H:%M:%S) start $unit load $(cut -d' ' -f1 /proc/loadavg) $(powerprofilesctl get)"
  WORKTREE="$worktree" "$here/run.sh" $unit $rest >/dev/null
  sleep 2
  while systemctl --user is-active -q "bnb-10-$unit"; do sleep 5; done
  echo "$(date +%H:%M:%S) done $unit $(systemctl --user show -p Result --value bnb-10-$unit)"
done
