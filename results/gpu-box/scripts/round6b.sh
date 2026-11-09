#!/bin/bash
# Round 6b (the author, 10:00 KST): coupled taus on the hidden-physics base, without the fade
# first (tau 1.25 s, 10 s: both extrapolation_decay and physics_hidden_decay), then tau 0.5 with
# the fade (physics_hidden_fade: the physics change on hidden rows scaled by exp(-age/tau)),
# then tournament 4: every 30M policy on C-fixed's base, both seed pairs in.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
until grep -q "round6 seed1 done" "$LOG"; do sleep 30; done

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
train $P-tau1.25 --set yemong.extrapolation_decay=1.25 --set yemong.physics_hidden_decay=1.25
# tau 10: the laptop's if it took it (marker runs/laptop-tau10, written when it confirms)
[ -f runs/laptop-tau10 ] || train $P-tau10 --set yemong.extrapolation_decay=10.0 --set yemong.physics_hidden_decay=10.0
# the fade: only once its build is merged (the switch must exist in this tree); wait up to 3 h
for _ in $(seq 36); do
  .venv/bin/bnb train "${COMMON[@]}" --set yemong.physics_hidden_decay=0.5 \
    --set yemong.physics_hidden_fade=true --print-config > /dev/null 2>&1 && break
  say "waiting for physics_hidden_fade to be merged"; sleep 300
done
train $P-fade --set yemong.physics_hidden_decay=0.5 --set yemong.physics_hidden_fade=true
# the author, 10:30 KST: two more (each the laptop's if it claims it: runs/laptop-<arm>)
[ -f runs/laptop-drsplit-seed1 ] || train 2026-11-09-bc-deadreckon-split-seed1 --seed 1
[ -f runs/laptop-persist-split-hp ] || train 2026-11-09-bc-persist-split-hiddenphysics \
  --set yemong.delta_baseline=persistence --set yemong.physics_hidden_decay=0.5
# the author, 10:55 KST: fill the wait for the laptop with absolute's 60M resume (the 60M
# round's baseline); its local checkpoints were deleted for disk, so take 30.4M from the hub
A=2026-11-09-bc-absolute
if ! grep -q '"status": "finished"' runs/$A/run.json 2>/dev/null || [ ! -d runs/$A/policies/step_60817408 ]; then
  /venv/main/bin/python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('julianzia/boost-and-broadside-runs', local_dir='runs/hub',
                  allow_patterns=['runs/$A/checkpoints/step_30408704/*'], token=os.environ['HF_TOKEN'])"
  mkdir -p runs/$A/checkpoints && cp -r runs/hub/runs/$A/checkpoints/step_30408704 runs/$A/checkpoints/
  say "start resume $A to 60M"
  if .venv/bin/bnb train --resume $A --bc-budget-steps 60000000 >> runs/$A.out 2>&1; then
    say "done resume $A"
  else
    say "FAILED resume $A (runs/$A.out)"
  fi
  docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
fi
say "round6b training done"

P30=policies/step_30408704/uniform_average
fetch() {  # marker, run: wait for the laptop's run on the hub; take its policy files
  [ -f "runs/$1" ] || return 0
  until /venv/main/bin/python runs/hub_fetch.py "$2"; do sleep 120; done
  mkdir -p runs/$2 && cp -r runs/hub/runs/$2/. runs/$2/ && say "downloaded $2 from the hub"
}
fetch laptop-tau10 $P-tau10-laptop
fetch laptop-persist-split-hp 2026-11-09-bc-persist-split-hiddenphysics-laptop
players=()
for run in 2026-11-09-bc-absolute 2026-11-09-bc-absolute-seed1 2026-11-09-bc-deadreckon-split \
    2026-11-09-bc-deadreckon-split-fourier 2026-11-09-bc-deadreckon-split-fastdecay-laptop \
    2026-11-09-bc-persist-split-laptop $P $P-seed1 $P-tau1.25 $P-tau10 $P-tau10-laptop $P-fade \
    2026-11-09-bc-deadreckon-split-seed1 2026-11-09-bc-persist-split-hiddenphysics \
    2026-11-09-bc-persist-split-hiddenphysics-laptop; do
  path=runs/$run/$P30
  # absolute was rewound to 23.1M for its 60M resume: its original 30.4M policy (the one
  # rated in tournament 3) moved to policies/rewound_1/
  [ $run = 2026-11-09-bc-absolute ] && path=runs/$run/policies/rewound_1/step_30408704/uniform_average
  [ -f $path/checkpoint.json ] && players+=($path) || say "tournament4: $run missing"
done
say "tournament4 start (${#players[@]} players)"
if .venv/bin/bnb ratings 2026-11-09-bc-absolute --no-league --no-ladder --with "${players[@]}" \
    --presets frontline deathmatch --out runs/ratings/round4 >> runs/ratings-round4.out 2>&1; then
  say "tournament4 done"
else
  say "tournament4 FAILED (runs/ratings-round4.out)"
fi
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
