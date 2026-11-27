#!/bin/bash
# Locked skill jobs (each about 8-10 minutes) until both setups reach the target error.
D=docs/rewrite/sections/lookahead-personas
cd /home/vizia/avesnova/boost-and-broadside/v2-defaults
S="uv run --no-sync python $D/skill.py /home/vizia/avesnova/boost-and-broadside/data/lookahead-personas/games.sqlite play --target-error 0.015 --round-games 16384"
LOGS=/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas/logs
for i in $(seq 1 12); do
  done1=$(cat $LOGS/skill*.log 2>/dev/null | grep -c "^1v1: done")
  done4=$(cat $LOGS/skill*.log 2>/dev/null | grep -c "^4v4: done")
  [ "$done1" -gt 0 ] && [ "$done4" -gt 0 ] && break
  cmd=""
  [ "$done1" -eq 0 ] && cmd="$S --setup 1v1 --minutes 2"
  [ "$done4" -eq 0 ] && cmd="${cmd:+$cmd; }$S --setup 4v4 --minutes 5"
  $D/locked.sh skill$i "$cmd"
done
