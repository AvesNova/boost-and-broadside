#!/bin/bash
# Round 7 (the author, 17:00 KST): tournament 4's front runners to 60M (the top four were within
# 24 on the mean): hiddenphysics (from its 43M checkpoint), -tau1.25 and deadreckon-split on the
# box; -tau10 on the laptop. Then tournament 5 at 60M with absolute's 60M as the null baseline and
# the same arms' 30M policies, reusing tournaments 3 and 4's games (--reuse), default layout.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
B=2026-11-09-bc
for name in $B-deadreckon-split-hiddenphysics $B-deadreckon-split-hiddenphysics-tau1.25 \
    $B-deadreckon-split; do
  if [ -d runs/$name/policies/step_60817408 ]; then say "$name already at 60M"; continue; fi
  say "start resume $name to 60M"
  if .venv/bin/bnb train --resume $name --bc-budget-steps 60000000 >> runs/$name.out 2>&1; then
    say "done resume $name"
  else
    say "FAILED resume $name (runs/$name.out)"
  fi
  docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
done
say "round7 box done"

# the laptop's tau10 at 60M (marker runs/laptop-tau10-60m once it confirms): fetch its policies
L=$B-deadreckon-split-hiddenphysics-tau10-laptop
P60=policies/step_60817408/uniform_average
if [ -f runs/laptop-tau10-60m ]; then
  until /venv/main/bin/python - "$L" <<'EOF'
import os, sys
from huggingface_hub import HfApi, snapshot_download
run = sys.argv[1]; repo = "julianzia/boost-and-broadside-runs"; tok = os.environ["HF_TOKEN"]
if f"runs/{run}/policies/step_60817408/uniform_average/checkpoint.json" not in HfApi(token=tok).list_repo_files(repo):
    sys.exit(1)
snapshot_download(repo, local_dir="runs/hub", allow_patterns=[f"runs/{run}/policies/**", f"runs/{run}/run.json",
                  f"runs/{run}/config.json", f"runs/{run}/metrics.jsonl"], token=tok)
EOF
  do sleep 120; done
  cp -r runs/hub/runs/$L/. runs/$L/ && say "downloaded $L at 60M"
fi

P30=policies/step_30408704/uniform_average
players=(runs/$B-absolute/policies/rewound_1/step_30408704/uniform_average runs/$B-absolute/$P60)
for run in $B-deadreckon-split-hiddenphysics $B-deadreckon-split-hiddenphysics-tau1.25 \
    $B-deadreckon-split $L; do
  players+=(runs/$run/$P30)
  [ -f runs/$run/$P60/checkpoint.json ] && players+=(runs/$run/$P60) || say "tournament5: $run at 60M missing"
done
say "tournament5 start (${#players[@]} players, reusing rounds 3 and 4)"
if .venv/bin/bnb ratings $B-absolute --no-league --no-ladder --with "${players[@]}" \
    --presets frontline deathmatch --reuse runs/ratings/round3 runs/ratings/round4 \
    --out runs/ratings/round5 >> runs/ratings-round5.out 2>&1; then
  say "tournament5 done"
else
  say "tournament5 FAILED (runs/ratings-round5.out)"
fi
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
