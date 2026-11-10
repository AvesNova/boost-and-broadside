#!/bin/bash
# Round 8b (the author, 21:20 KST): before RL, compare the 120M live, uniform-average and
# exponential-average policies of the champion and the null: tournament 6 (six players), then a
# scorecard per run (runs/scorecard.py: each variant on fresh games, nothing trained). RL's
# starting point (one kind for both) is chosen from these; this runner does not start RL.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
while pgrep -f "resume 2026-11-09-bc-absolute --bc" > /dev/null; do sleep 30; done
B=2026-11-09-bc
grep -q '"status": "finished"' runs/$B-absolute/run.json || { say "absolute's 120M BC did not finish; stopping"; exit 1; }
say "done resume $B-absolute to 120M"
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1

last() { ls -d runs/$1/policies/step_* | sort -V | tail -1; }
players=()
for run in $B-deadreckon-split-hiddenphysics $B-absolute; do
  s=$(last $run)
  for v in live uniform_average exponential_average; do players+=($s/$v); done
done
say "tournament6 start (${#players[@]} players: 120M live / uniform / exponential, champion and null)"
if .venv/bin/bnb ratings $B-absolute --no-league --no-ladder --with "${players[@]}" \
    --presets frontline deathmatch --out runs/ratings/round6 >> runs/ratings-round6.out 2>&1; then
  say "tournament6 done"
else
  say "tournament6 FAILED (runs/ratings-round6.out)"
fi
for run in $B-deadreckon-split-hiddenphysics $B-absolute; do
  say "start scorecard $run"
  if .venv/bin/python runs/scorecard.py runs/$run runs/scorecard-$run.json >> runs/scorecard-$run.out 2>&1; then
    say "done scorecard $run"
  else
    say "FAILED scorecard $run (runs/scorecard-$run.out)"
  fi
done
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
say "round8b done: choose RL's starting point"
