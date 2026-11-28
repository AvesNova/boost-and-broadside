#!/bin/bash
# Item 20 step 1 (GPU, one job): mirror-game statistics of every variant at 4v4 and 1v1
# (one JSON per variant and size, skipped if present), then a round robin of all variants at
# 4v4 and 1v1 in one lean.eval store (every game stored as it ends), its Elo and styles.
# Run under flock data/gpu.lock as a systemd user unit (bnb-20-step1).
set -u
cd /home/vizia/avesnova/boost-and-broadside/v2-trigger20
HERE=docs/rewrite/sections/trigger-20
OUT=${OUT:-/home/vizia/avesnova/boost-and-broadside/data/trigger-20}
VARIANTS=${VARIANTS:-"old-t1 old-t3 old-t6 old-t10 new-k0 new-k1 new-k2 new-k4"}
MIRROR_4=${MIRROR_4:-2048}
MIRROR_1=${MIRROR_1:-4096}
BUDGET_4=${BUDGET_4:-40000}
BUDGET_1=${BUDGET_1:-60000}
export PYTHONPATH=.
mkdir -p "$OUT/mirror"
echo "$(date +%H:%M:%S) start; power $(powerprofilesctl get); load $(cut -d' ' -f1-3 /proc/loadavg)"
for v in $VARIANTS; do
  for size in 4 1; do
    target="$OUT/mirror/${v}_${size}v${size}.json"
    [ -f "$target" ] && continue
    envs=$MIRROR_4; [ "$size" = 1 ] && envs=$MIRROR_1
    echo "$(date +%H:%M:%S) mirror $v ${size}v${size} ($envs games)"
    uv run --no-sync python $HERE/behaviour.py --agent $HERE/variants/$v.json --size $size \
      --envs $envs --device cuda --out "$OUT/mirror" | tail -1 || echo "FAILED mirror $v $size"
  done
done
agents=""
for v in $VARIANTS; do agents="$agents scripted:$HERE/variants/$v.json"; done
for size in 4 1; do
  budget=$BUDGET_4; [ "$size" = 1 ] && budget=$BUDGET_1
  echo "$(date +%H:%M:%S) round robin ${size}v${size} (budget $budget)"
  uv run --no-sync python -m lean.eval play "$OUT/elo.sqlite" $agents --sizes $size $size \
    --target-error 0.02 --budget $budget --envs 2048 || echo "FAILED round robin $size"
  uv run --no-sync python -m lean.eval elo "$OUT/elo.sqlite" --sizes $size $size \
    > "$OUT/elo_${size}v${size}.txt"
  uv run --no-sync python -m lean.eval styles "$OUT/elo.sqlite" $agents --sizes $size $size \
    > "$OUT/styles_${size}v${size}.txt"
  cat "$OUT/elo_${size}v${size}.txt"
done
echo "$(date +%H:%M:%S) done; load $(cut -d' ' -f1-3 /proc/loadavg)"
