#!/bin/bash
# Extra games at seed 2 for the averaged and final live checkpoints only: +3,072 per point
# at 8 and 16 ships, +2,048 at 32 ships (4 ships came from run_extra_laptop.sh).
# Run from anywhere; appends to $FAN_DIR/fan_731_n{N}.extra.jsonl (default out/studies/fan).
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE/../../.."
export VIRTUAL_ENV=
F=${FAN_DIR:-out/studies/fan}; mkdir -p "$F"
LOCK=${GPU_LOCK:-out/gpu.lock}
D=checkpoints/wild-universe-731
for spec in "8 3072" "16 3072" "32 2048"; do
  set -- $spec; n=$1; g=$2
  opp="$n $((n*5/4)) $((n*3/2)) $((n*7/4)) $((n*2))"
  for c in $D/avg_step_000671846400.pt $D/step_000671846400.pt; do
    echo "== n$n $c $(date +%H:%M:%S)"
    flock "$LOCK" systemd-run --user --scope -q -p MemoryMax=10G \
      uv run --no-sync python "$HERE/fan_eval.py" $c $F/fan_731_n$n.extra.jsonl --trained $n --scripted $opp --games $g --seed 2 2>&1 | grep -E "^[0-9]+v|Error|Traceback"
  done
  echo "SIZE_DONE n$n $(date +%H:%M:%S)"
done
echo ALL_DONE
