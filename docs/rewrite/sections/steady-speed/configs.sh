#!/bin/bash
# configs.sh TREE OUT ENVIRONMENTS [extra --set lines...]: the launch config (launch.toml's
# lineup and shared --set lines, preset real_scale_all_modes) resolved per stage into
# OUT/{bc,ppo}.json with tree TREE's code (after: fix/learner-recompile, whose defaults are
# the champion's model; before: rel/candidate 9f8c20fc, given the champion's five lines as
# its launch.toml gives them).
set -euo pipefail
TREE=$1; OUT=$2; ENVIRONMENTS=$3; shift 3
mkdir -p "$OUT"
SETS=(--set host_store=false --set policy_snapshot_every=5000000
  --set 'lineup=["Yongma","Yongma","Jujak","Jujak","Jujak"]')
if [ "$TREE" = before ]; then
  SETS+=(--set yemong.next_state=delta --set yemong.delta_baseline=extrapolation
    --set yemong.delta_split=true --set yemong.physics_hidden_decay=0.5
    --set learner.next_state=0.0033)
fi
for extra in "$@"; do SETS+=(--set "$extra"); done
for objective in bc ppo; do
  PYTHONPATH=/workspace/wt-$TREE/src /workspace/bnb/.venv/bin/bnb train --preset real_scale_all_modes \
    --environments "$ENVIRONMENTS" --objective $objective --update-policies False "${SETS[@]}" \
    --print-config > "$OUT/$objective.json"
done
