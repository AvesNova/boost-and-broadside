#!/bin/bash
# Round 2 finished on the rented 3090: C-fixed (the reference, retrained here), arms 2d, 3d,
# 4d to 30M each, and each arm's tournament. One run at a time. Log: runs/arms.log.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BNB=.venv/bin/bnb
LOG=runs/arms.log
SPEED="--micro-batch-rows 2560 --set host_store=false"
BASE="--preset mixed --objective bc --bc-budget-steps 30000000 --set yemong.order=temporal_first"
DELTA="--set yemong.next_state=delta --set yemong.delta_baseline=persistence --set yemong.spread_inputs=scalar --set learner.next_state=0.0033"

say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }

train() {  # name, settings...
  local name=$1; shift
  if [ -f "runs/$name/run.json" ] && grep -q '"status": "finished"' "runs/$name/run.json"; then
    say "$name already finished"; return 0
  fi
  say "start $name"
  if $BNB train --name "$name" $BASE $SPEED "$@" >> "runs/$name.out" 2>&1; then
    say "done $name"; return 0
  fi
  say "FAILED $name (see runs/$name.out); retrying with host_store=true"
  mv "runs/$name" "runs/$name-failed-$(date +%s)" 2>/dev/null
  if $BNB train --name "$name" $BASE --micro-batch-rows 2560 "$@" >> "runs/$name.out" 2>&1; then
    say "done $name (host_store=true)"; return 0
  fi
  say "FAILED $name again"; return 1
}

tournament() {  # out name, run, players...
  local out=$1 run=$2; shift 2
  local players=()
  for p in "$@"; do  # the newest uniform average (30.4M unless a stop rule fired earlier)
    players+=("$(ls -d runs/$p/policies/step_*/uniform_average | sort -V | tail -1)")
  done
  say "tournament $out: $*"
  if $BNB ratings "$run" --no-league --with "${players[@]}" --presets frontline deathmatch \
      --out "runs/ratings/$out" >> "runs/ratings-$out.out" 2>&1; then
    say "tournament $out done"
  else
    say "tournament $out FAILED (see runs/ratings-$out.out)"
  fi
}

mkdir -p runs/ratings
C=2026-11-08-bc-arm-C-fixed-3090
A2=2026-11-08-bc-arm2d-persistence-3090
A3=2026-11-08-bc-arm3d-split-3090
A4=2026-11-08-bc-arm4d-extrapolation-3090

train $C
train $A2 $DELTA && tournament arm2d $A2 $A2 $C
train $A3 $DELTA --set yemong.delta_split=true --set learner.correction=0.0033 \
  && tournament arm3d $A3 $A3 $A2 $C
train $A4 $DELTA --set yemong.delta_split=true --set learner.correction=0.0033 \
  --set yemong.delta_baseline=extrapolation \
  && tournament arm4d $A4 $A4 $A3 $C
say "all done"
