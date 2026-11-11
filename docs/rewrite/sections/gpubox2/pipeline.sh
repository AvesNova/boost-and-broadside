#!/bin/bash
# One-head A/B (2026-11-12): the champion's pipeline repeated exactly on its own code
# (src = 6af08e09 = 09db261c, the champion's final BC and first RL commit), changing only
# yemong.heads 2 -> 1. Stages, each skipped when done and resumed from its newest checkpoint
# when interrupted (the box is interruptible: supervisor restarts this driver on boot):
#   1. BC to 120.6M (budget 120,000,000: 115 updates), the champion's settings
#   2. RL 58 updates (60.8M) from the BC 120.6M uniform average, the champion's settings
#   3. RL resumed to 116 updates (121.6M), as the champion's second RL segment
#   4. RL resumed to 172 updates (180.4M), as the champion's third RL segment
#   5. the tournament (tournament.sh)
# Every event goes to runs/pipeline.log (UTC). Run as the supervisor program onehead.
set -u
REPO=/workspace/boost-and-broadside
cd "$REPO"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/pipeline.log
mkdir -p runs runs/failed
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
BNB=.venv/bin/bnb
BC=2026-11-12-bc-onehead
RL=2026-11-12-rl-onehead
BC_FINAL=120586240
RL60=60817408
RL121=121634816
RL180=180355072
INIT=runs/$BC/policies/step_$BC_FINAL/uniform_average

# The champion's BC command (its run.json, segment 0) with the 120M budget and one head;
# checkpoints every 4 updates (~6 min) so an interruption loses little.
BC_ARGS=(--preset mixed --objective bc --bc-budget-steps 120000000
  --micro-batch-rows 2560 --set host_store=false --set yemong.order=temporal_first
  --set "learner.discount=[0.991,0.995,0.995,0.999,1.0]"
  --set "learner.trace_decay=[0.99,0.99,0.99,0.99,1.0]"
  --set stop.score_target=1.0 --set stop.plateau_steps=1000000000
  --set yemong.next_state=delta --set yemong.spread_inputs=scalar --set learner.next_state=0.0033
  --set yemong.delta_baseline=extrapolation --set yemong.delta_split=true
  --set learner.correction=0.0033 --set yemong.physics_hidden_decay=0.5
  --set yemong.heads=1)
BC_EVERY=4194304
# The champion's RL command (its run.json, segment 0); the model's settings from the file.
# Micro-batch rows: memory and speed only (micro-batches add up to the minibatch). RL's
# minibatch is 2560 rows (32 minibatches of 256 views); the champion's 640 ran out of memory
# on this code (gpu-box-log 01:00 KST), so 640 is tried first and 320 (the champion's) after
# an out-of-memory failure (runs/rl-rows records the choice).
RL_ROWS=$(cat runs/rl-rows 2>/dev/null || echo 640)
RL_ARGS=(--init-from $INIT --preset mixed --updates 58 --micro-batch-rows $RL_ROWS
  --set host_store=false)
RL_EVERY=3145728

finished() { grep -q '"status": "finished"' "runs/$1/run.json" 2>/dev/null; }
has_checkpoint() { ls -d runs/$1/checkpoints/step_* > /dev/null 2>&1; }

# A record left "running" by a killed process (a box stop): after a restart the recorded pid
# may belong to another process, which `bnb train --resume` would read as the run alive.
unstale() {
  local run=$1 pid
  pid=$(.venv/bin/python -c "import json;print(json.load(open('runs/$run/run.json')).get('pid') or 0)" 2>/dev/null)
  if grep -q '"status": "running"' "runs/$run/run.json" 2>/dev/null && \
     ! tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q "bnb train"; then
    .venv/bin/python - "$run" <<'PY'
import json, sys
path = f"runs/{sys.argv[1]}/run.json"
record = json.load(open(path))
record["status"] = "stopped"
record["message"] = "marked stopped by pipeline.sh: its process died (interruption)"
json.dump(record, open(path, "w"), indent=1)
PY
    say "$run: stale running record (pid $pid gone) marked stopped"
  fi
}

# Start or resume one stage; a run directory without a checkpoint is moved aside first.
stage() {  # name, budget check (a command), fresh args..., -- resume args...
  local name=$1 done_check=$2; shift 2
  local fresh=() resume=() seen=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then seen=1; continue; fi
    if [ $seen = 0 ]; then fresh+=("$a"); else resume+=("$a"); fi
  done
  for attempt in 1 2 3; do
    if $done_check; then say "$name: stage done"; return 0; fi
    if [ -d runs/$name ] && ! has_checkpoint $name; then
      say "$name: no checkpoint; moved aside to runs/failed, starting fresh"
      mv runs/$name runs/failed/$name-$(date +%s)
    fi
    if [ -d runs/$name ]; then
      unstale $name
      say "$name: resume (attempt $attempt) from $(ls -d runs/$name/checkpoints/step_* | sort -V | tail -1)"
      $BNB train --resume $name "${resume[@]}" >> runs/$name.out 2>&1
    else
      say "$name: fresh start (attempt $attempt)"
      $BNB train --name $name "${fresh[@]}" >> runs/$name.out 2>&1
    fi
    say "$name: process exited with $?"
    # RL at 640 rows out of memory: the next fresh attempt takes 320 rows
    if [ "$name" = "$RL" ] && [ "$RL_ROWS" = 640 ] && ! has_checkpoint $name && \
        tail -50 runs/$name.out | grep -q "OutOfMemoryError\|out of memory\|MemoryBudgetError"; then
      say "$name: out of memory at 640 rows; fresh start at 320"
      echo 320 > runs/rl-rows; RL_ROWS=320
      for i in "${!fresh[@]}"; do [ "${fresh[$i]}" = 640 ] && fresh[$i]=320; done
    fi
  done
  $done_check && return 0
  say "$name: FAILED after 3 attempts (runs/$name.out); the pipeline stops"
  return 1
}

newest() {  # the newest checkpoint's step of run $1 (0 if none)
  ls -d runs/$1/checkpoints/step_* 2>/dev/null | sed 's/.*step_//' | sort -n | tail -1 | grep . || echo 0
}
bc_done() { [ "$(newest $BC)" -ge $BC_FINAL ] && [ -f $INIT/checkpoint.json ]; }
rl60_done() { [ "$(newest $RL)" -ge $RL60 ] && [ -f runs/$RL/policies/step_$RL60/uniform_average/checkpoint.json ]; }
rl121_done() { [ "$(newest $RL)" -ge $RL121 ] && [ -f runs/$RL/policies/step_$RL121/uniform_average/checkpoint.json ]; }
rl180_done() { [ "$(newest $RL)" -ge $RL180 ] && [ -f runs/$RL/policies/step_$RL180/uniform_average/checkpoint.json ]; }

say "pipeline start (pid $$, host $(hostname), commit $(git rev-parse --short HEAD))"
stage $BC bc_done "${BC_ARGS[@]}" --checkpoint-every $BC_EVERY -- --checkpoint-every $BC_EVERY || exit 2
if ! rl60_done; then
  [ -f runs/rl-config-checked ] || { docs/rewrite/sections/gpubox2/check_config.sh rl || exit 2; }
fi
stage $RL rl60_done "${RL_ARGS[@]}" --checkpoint-every $RL_EVERY -- --checkpoint-every $RL_EVERY || exit 2
stage $RL rl121_done -- --updates 116 --checkpoint-every $RL_EVERY || exit 2
stage $RL rl180_done -- --updates 172 --checkpoint-every $RL_EVERY || exit 2
[ -f runs/tournament.done ] || docs/rewrite/sections/gpubox2/tournament.sh || exit 2
say "pipeline done"
