#!/bin/bash
# Round 4, part b: while the laptop's second arm finishes (~08:45 KST), absolute at seed 1 (the
# seed spread on this box, for the 60M "cannot separate" call); then tournament 3 once the
# laptop's two runs are on the hub (downloaded: their policy files only).
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
until grep -q "round4 training done" "$LOG"; do sleep 30; done

NAME=2026-11-09-bc-absolute-seed1
say "start $NAME"
if .venv/bin/bnb train --name $NAME --seed 1 --preset mixed --objective bc \
    --bc-budget-steps 30000000 --micro-batch-rows 2560 --set host_store=false \
    --set yemong.order=temporal_first \
    --set "learner.discount=[0.991,0.995,0.995,0.999,1.0]" \
    --set "learner.trace_decay=[0.99,0.99,0.99,0.99,1.0]" \
    --set stop.score_target=1.0 --set stop.plateau_steps=1000000000 >> runs/$NAME.out 2>&1; then
  say "done $NAME"
else
  say "FAILED $NAME (runs/$NAME.out)"
fi
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1

# the laptop's runs: wait for their final uniform averages on the hub, download the policies
LAPTOP=(2026-11-09-bc-deadreckon-split-fastdecay-laptop 2026-11-09-bc-persist-split-laptop)
P=policies/step_30408704/uniform_average
for run in "${LAPTOP[@]}"; do
  until /venv/main/bin/python - "$run" <<'EOF'
import os, sys
from huggingface_hub import HfApi, snapshot_download
run = sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
files = api.list_repo_files("julianzia/boost-and-broadside-runs")
want = f"runs/{run}/policies/step_30408704/uniform_average/checkpoint.json"
if want not in files:
    sys.exit(1)
snapshot_download("julianzia/boost-and-broadside-runs", local_dir="runs/hub",
                  allow_patterns=[f"runs/{run}/policies/*", f"runs/{run}/policies/**",
                                  f"runs/{run}/run.json", f"runs/{run}/config.json",
                                  f"runs/{run}/metrics.jsonl"], token=os.environ["HF_TOKEN"])
EOF
  do sleep 120; done
  mkdir -p runs/$run && cp -rn runs/hub/runs/$run/. runs/$run/
  say "downloaded $run from the hub"
done
say "round4b ready for the tournament"
