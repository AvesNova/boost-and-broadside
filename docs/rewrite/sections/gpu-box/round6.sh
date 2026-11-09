#!/bin/bash
# Round 6 (the author, 09:30 KST): after round 5's box resumes, deadreckon-split-hiddenphysics
# without decay (tau 1e9 s: every hidden row weighs 1) to 30M; persist-split to 60M on the box
# unless the laptop took it (marker runs/laptop-persist-60m); then tournament 4 with everything:
# every 30M uniform average from tournament 3 again (identical policies: the tournament's own
# variance, next to the seed variance of absolute's two seeds), the new arm, and the 60M ones.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
until grep -q "round5 box done" "$LOG"; do sleep 30; done

NAME=2026-11-09-bc-deadreckon-split-hiddenphysics-nodecay
if ! grep -q '"status": "finished"' runs/$NAME/run.json 2>/dev/null; then
  say "start $NAME"
  if .venv/bin/bnb train --name $NAME --preset mixed --objective bc --bc-budget-steps 30000000 \
      --micro-batch-rows 2560 --set host_store=false --set yemong.order=temporal_first \
      --set "learner.discount=[0.991,0.995,0.995,0.999,1.0]" \
      --set "learner.trace_decay=[0.99,0.99,0.99,0.99,1.0]" \
      --set stop.score_target=1.0 --set stop.plateau_steps=1000000000 \
      --set yemong.next_state=delta --set yemong.spread_inputs=scalar --set learner.next_state=0.0033 \
      --set yemong.delta_baseline=extrapolation --set yemong.delta_split=true \
      --set learner.correction=0.0033 --set yemong.physics_hidden_decay=1e9 >> runs/$NAME.out 2>&1; then
    say "done $NAME"
  else
    say "FAILED $NAME (runs/$NAME.out)"
  fi
  docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
fi

PS=2026-11-09-bc-persist-split-laptop
P60=policies/step_60817408/uniform_average
if [ -f runs/laptop-persist-60m ]; then
  say "persist-split 60M is the laptop's: waiting for it on the hub"
  until /venv/main/bin/python - <<EOF
import os, sys
from huggingface_hub import HfApi, snapshot_download
api = HfApi(token=os.environ["HF_TOKEN"])
if "runs/$PS/$P60/checkpoint.json" not in api.list_repo_files("julianzia/boost-and-broadside-runs"):
    sys.exit(1)
snapshot_download("julianzia/boost-and-broadside-runs", local_dir="runs/hub",
                  allow_patterns=["runs/$PS/policies/**", "runs/$PS/run.json", "runs/$PS/config.json",
                                  "runs/$PS/metrics.jsonl"], token=os.environ["HF_TOKEN"])
EOF
  do sleep 120; done
  cp -r runs/hub/runs/$PS/. runs/$PS/ && say "downloaded $PS at 60M"
else
  say "start resume $PS to 60M on the box (checkpoint from the hub)"
  /venv/main/bin/python - <<EOF
import os
from huggingface_hub import snapshot_download
snapshot_download("julianzia/boost-and-broadside-runs", local_dir="runs/hub",
                  allow_patterns=["runs/$PS/checkpoints/step_30408704/*"], token=os.environ["HF_TOKEN"])
EOF
  mkdir -p runs/$PS/checkpoints && cp -r runs/hub/runs/$PS/checkpoints/step_30408704 runs/$PS/checkpoints/
  if .venv/bin/bnb train --resume $PS --bc-budget-steps 60000000 >> runs/$PS.out 2>&1; then
    say "done resume $PS"
  else
    say "FAILED resume $PS (runs/$PS.out)"
  fi
  docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
fi

P30=policies/step_30408704/uniform_average
players=()
for run in 2026-11-09-bc-absolute 2026-11-09-bc-absolute-seed1 2026-11-09-bc-deadreckon-split \
    2026-11-09-bc-deadreckon-split-fourier 2026-11-09-bc-deadreckon-split-hiddenphysics \
    2026-11-09-bc-deadreckon-split-fastdecay-laptop 2026-11-09-bc-persist-split-laptop \
    2026-11-09-bc-deadreckon-split-hiddenphysics-nodecay; do
  [ -f runs/$run/$P30/checkpoint.json ] && players+=(runs/$run/$P30) || say "tournament4: $run at 30M missing"
done
for run in 2026-11-09-bc-deadreckon-split-hiddenphysics 2026-11-09-bc-deadreckon-split $PS; do
  [ -f runs/$run/$P60/checkpoint.json ] && players+=(runs/$run/$P60) || say "tournament4: $run at 60M missing"
done
say "tournament4 start (${#players[@]} players)"
if .venv/bin/bnb ratings 2026-11-09-bc-absolute --no-league --no-ladder --with "${players[@]}" \
    --presets frontline deathmatch --out runs/ratings/round4 >> runs/ratings-round4.out 2>&1; then
  say "tournament4 done"
else
  say "tournament4 FAILED (runs/ratings-round4.out)"
fi
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
