#!/bin/bash
# Item 13 GPU checks after the behaviour unit, each under the GPU lock in this unit:
# the real trainer (static compile, fail_on_recompile) for 2 BC + 2 RL updates, then the cold
# compile check (budget 175 s) once the CPU is quiet.   gpu.sh DATA
set -u
DATA=$1
cd /home/vizia/avesnova/boost-and-broadside/v2-zone13
export PYTHONPATH=.
echo "trainer start $(date +%H:%M:%S) power $(powerprofilesctl get) load $(cut -d' ' -f1-3 /proc/loadavg) tree $(git rev-parse --short HEAD)"
uv run --no-sync python -m lean.train --bc-updates 2 --updates 2 --out "$DATA/train" || echo "TRAINER FAILED $?"
echo "trainer done $(date +%H:%M:%S)"
# Quiet CPU for the timed compile: wait for the load to settle below 4 (at most 10 min).
for i in $(seq 120); do awk '{exit !($1 < 4)}' /proc/loadavg && break; sleep 5; done
echo "compile start $(date +%H:%M:%S) power $(powerprofilesctl get) load $(cut -d' ' -f1-3 /proc/loadavg)"
uv run --no-sync python docs/rewrite/sections/scoreboard/compile_check.py || echo "COMPILE CHECK FAILED $?"
echo "compile done $(date +%H:%M:%S)"
