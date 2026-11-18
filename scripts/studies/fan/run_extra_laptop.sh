#!/bin/bash
# Extra games at seed 2: +3,072 per point for the averaged and final live checkpoints at
# 4 ships. (This launcher also covered the ladder and the larger sizes; it was stopped after
# 4 ships and one ladder checkpoint, and run_final_laptop.sh took over. merge_extra.py
# ignores extra rows of ladder checkpoints, so the ladder stays at its seed-1 games.)
# Run from anywhere; writes $FAN_DIR/fan_731_n{N}.extra.jsonl (default out/studies/fan).
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE/../../.."
export VIRTUAL_ENV=
F=${FAN_DIR:-out/studies/fan}; mkdir -p "$F"
LOCK=${GPU_LOCK:-out/gpu.lock}
D=checkpoints/wild-universe-731
for n in 4 8 16 32; do
  opp="$n $((n*5/4)) $((n*3/2)) $((n*7/4)) $((n*2))"
  for c in $D/avg_step_000671846400.pt $D/step_000671846400.pt $(ls -r $D/ladder_step_*.pt); do
    echo "== n$n $c $(date +%H:%M:%S)"
    flock "$LOCK" systemd-run --user --scope -q -p MemoryMax=10G \
      uv run --no-sync python "$HERE/fan_eval.py" $c $F/fan_731_n$n.extra.jsonl --trained $n --scripted $opp --games 3072 --seed 2 2>&1 | grep -E "^[0-9]+v|Error|Traceback"
  done
  echo "SIZE_DONE n$n $(date +%H:%M:%S)"
done
echo ALL_DONE
