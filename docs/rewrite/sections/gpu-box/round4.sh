#!/bin/bash
# Round 3 again, on the correct base (the author, 2026-11-09 03:30 KST): every arm with C-fixed's
# learner settings and the laptop arms' stop flags, 30M each, then one tournament. Starts only if
# the confirmation (runs/confirm.sh) shows positive outcome Brier skill.
# Not rerun: persist-onehead and absolute-bigbatch (failed clearly in tournament 1).
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }

until grep -q "done 2026-11-09-bc-absolute-cfixed-learner-check" "$LOG"; do sleep 30; done
mean=$(grep "done 2026-11-09-bc-absolute-cfixed-learner-check" "$LOG" | tail -1 | sed 's/.*| mean //')
if ! python3 -c "import sys; sys.exit(0 if float('$mean') > 0.05 else 1)" 2>/dev/null; then
  say "round4 NOT started: the confirmation's Brier mean is $mean (expected > +0.05); stopping"
  exit 1
fi
say "round4 start (confirmation Brier mean $mean)"

COMMON=(--preset mixed --objective bc --bc-budget-steps 30000000
  --micro-batch-rows 2560 --set host_store=false --set yemong.order=temporal_first
  --set "learner.discount=[0.991,0.995,0.995,0.999,1.0]"
  --set "learner.trace_decay=[0.99,0.99,0.99,0.99,1.0]"
  --set stop.score_target=1.0 --set stop.plateau_steps=1000000000)
DELTA=(--set yemong.next_state=delta --set yemong.spread_inputs=scalar --set learner.next_state=0.0033)
SPLIT=(--set yemong.delta_split=true --set learner.correction=0.0033)
DR=("${DELTA[@]}" --set yemong.delta_baseline=extrapolation "${SPLIT[@]}")
PS=("${DELTA[@]}" --set yemong.delta_baseline=persistence "${SPLIT[@]}")

train() {  # name, settings...
  local name=$1; shift
  if [ -f "runs/$name/run.json" ] && grep -q '"status": "finished"' "runs/$name/run.json"; then
    say "$name already finished"; return 0
  fi
  say "start $name"
  if .venv/bin/bnb train --name "$name" "${COMMON[@]}" "$@" >> "runs/$name.out" 2>&1; then
    say "done $name"
  else
    say "FAILED $name (runs/$name.out)"
  fi
  docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
}

P=2026-11-09-bc
train $P-absolute
train $P-deadreckon-split "${DR[@]}"
train $P-deadreckon-split-fourier "${DR[@]}" --set yemong.position_inputs=fourier
# hidden physics: only once its build is merged (the switch must exist in this tree)
if .venv/bin/bnb train "${COMMON[@]}" "${DR[@]}" --set yemong.physics_hidden_decay=0.5 \
    --print-config > /dev/null 2>&1; then
  train $P-deadreckon-split-hiddenphysics "${DR[@]}" --set yemong.physics_hidden_decay=0.5
else
  say "hiddenphysics skipped: yemong.physics_hidden_decay not in this tree yet"
fi
# The author's laptop may take these two (2026-11-09 03:45 KST plan): skip one whose run the
# laptop has claimed (a file runs/laptop-<arm> written when it starts).
[ -f runs/laptop-fastdecay ] || train $P-deadreckon-split-fastdecay "${DR[@]}" --set yemong.extrapolation_decay=0.2
[ -f runs/laptop-persist-split ] || train $P-persist-split "${PS[@]}"
say "round4 training done"
