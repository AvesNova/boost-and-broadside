#!/bin/bash
# configs.sh OUT ENVIRONMENTS [extra --set lines...]: the launch config (launch.toml's shared
# --set lines, preset real_scale_all_modes) resolved per stage into OUT/{bc,ppo}.json, the
# measurement's stand-in lineup (feat/launch has no Yongma/Jujak yet; same slots, same shapes).
set -euo pipefail
OUT=$1; ENVIRONMENTS=$2; shift 2
mkdir -p "$OUT"
LINEUP='["vizia","vizia","Spitfire IX","Spitfire IX","Spitfire IX"]'
SETS=(
  --set yemong.next_state=delta --set yemong.delta_baseline=extrapolation
  --set yemong.delta_split=true --set yemong.physics_hidden_decay=0.5
  --set learner.next_state=0.0033 --set host_store=false
  --set policy_snapshot_every=5000000 --set "lineup=$LINEUP"
)
for extra in "$@"; do SETS+=(--set "$extra"); done
for objective in bc ppo; do
  .venv/bin/bnb train --preset real_scale_all_modes --environments "$ENVIRONMENTS" \
    --objective $objective --update-policies False "${SETS[@]}" --print-config > "$OUT/$objective.json"
done
