#!/bin/bash
# Round 6 (the author, 09:40 KST): the 60M resumes stopped for now; three more 30M arms on the
# hidden-physics base, then one tournament of every 30M policy on C-fixed's base, both seed pairs
# included (absolute / absolute-seed1, hiddenphysics / hiddenphysics-seed1) to show variance.
#  - nodecay: every hidden row weighs 1 in the physics loss (physics_hidden_decay 1e9 s).
#  - slowdecay: the held rates fade 2.5x slower (extrapolation_decay 0.5 -> 1.25 s; fastdecay's
#    0.2 s hurt), with hidden physics.
#  - hiddenphysics-seed1: the leader at a second seed.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
# the hiddenphysics resume stops after its current update (STOP file)
while pgrep -f "resume 2026-11-09-bc-deadreckon-split-hiddenphysics " > /dev/null; do sleep 10; done

COMMON=(--preset mixed --objective bc --bc-budget-steps 30000000
  --micro-batch-rows 2560 --set host_store=false --set yemong.order=temporal_first
  --set "learner.discount=[0.991,0.995,0.995,0.999,1.0]"
  --set "learner.trace_decay=[0.99,0.99,0.99,0.99,1.0]"
  --set stop.score_target=1.0 --set stop.plateau_steps=1000000000
  --set yemong.next_state=delta --set yemong.spread_inputs=scalar --set learner.next_state=0.0033
  --set yemong.delta_baseline=extrapolation --set yemong.delta_split=true
  --set learner.correction=0.0033)
train() {  # name, settings...
  local name=$1; shift
  if grep -q '"status": "finished"' runs/$name/run.json 2>/dev/null; then say "$name already finished"; return; fi
  say "start $name"
  if .venv/bin/bnb train --name "$name" "${COMMON[@]}" "$@" >> "runs/$name.out" 2>&1; then
    say "done $name"
  else
    say "FAILED $name (runs/$name.out)"
  fi
  docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
}
P=2026-11-09-bc-deadreckon-split-hiddenphysics
train $P-nodecay --set yemong.physics_hidden_decay=1e9
train $P-slowdecay --set yemong.physics_hidden_decay=0.5 --set yemong.extrapolation_decay=1.25
train $P-seed1 --seed 1 --set yemong.physics_hidden_decay=0.5
say "round6 training done"

P30=policies/step_30408704/uniform_average
players=()
for run in 2026-11-09-bc-absolute 2026-11-09-bc-absolute-seed1 2026-11-09-bc-deadreckon-split \
    2026-11-09-bc-deadreckon-split-fourier 2026-11-09-bc-deadreckon-split-fastdecay-laptop \
    2026-11-09-bc-persist-split-laptop $P $P-seed1 $P-nodecay $P-slowdecay; do
  [ -f runs/$run/$P30/checkpoint.json ] && players+=(runs/$run/$P30) || say "tournament4: $run missing"
done
say "tournament4 start (${#players[@]} players)"
if .venv/bin/bnb ratings 2026-11-09-bc-absolute --no-league --no-ladder --with "${players[@]}" \
    --presets frontline deathmatch --out runs/ratings/round4 >> runs/ratings-round4.out 2>&1; then
  say "tournament4 done"
else
  say "tournament4 FAILED (runs/ratings-round4.out)"
fi
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
