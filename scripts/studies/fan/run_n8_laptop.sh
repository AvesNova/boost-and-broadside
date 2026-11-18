#!/bin/bash
# 8-ship sweep (seed 1, 1,024 games per point): the final live and averaged checkpoints
# first, then the frozen ladder, newest first. One locked GPU job per checkpoint.
# Run from anywhere; writes $FAN_DIR/fan_731_n8.jsonl (default out/studies/fan).
set -e
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE/../../.."
export VIRTUAL_ENV=
F=${FAN_DIR:-out/studies/fan}; mkdir -p "$F"
LOCK=${GPU_LOCK:-out/gpu.lock}
OUT=$F/fan_731_n8.jsonl
D=checkpoints/wild-universe-731
for c in $D/step_000671846400.pt $D/avg_step_000671846400.pt $(ls -r $D/ladder_step_*.pt); do
  echo "== $c $(date +%H:%M:%S)"
  flock "$LOCK" systemd-run --user --scope -q -p MemoryMax=10G \
    uv run --no-sync python "$HERE/fan_eval.py" $c $OUT --trained 8 --scripted 8 10 12 14 16 --games 1024 2>&1 | grep -E "^8v|Error|Traceback"
done
echo DONE $(date +%H:%M:%S)
