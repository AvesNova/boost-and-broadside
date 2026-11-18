#!/bin/bash
# 16- then 32-ship sweeps (seed 1, 1,024 games per point) on a rented GPU box: the final
# live and averaged checkpoints first, then the frozen ladder. One JSON line per point.
# Run from anywhere; writes $FAN_DIR/fan_731_n{16,32}.jsonl (default out/studies/fan).
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE/../../.."
F=${FAN_DIR:-out/studies/fan}; mkdir -p "$F"
D=checkpoints/wild-universe-731
for N in 16 32; do
  OUT=$F/fan_731_n$N.jsonl
  S="$N $((N*5/4)) $((N*3/2)) $((N*7/4)) $((N*2))"
  for c in $D/step_000671846400.pt $D/avg_step_000671846400.pt $(ls -r $D/ladder_step_*.pt); do
    echo "== N=$N $c $(date +%H:%M:%S)"
    env -u VIRTUAL_ENV .venv/bin/python "$HERE/fan_eval.py" $c $OUT --trained $N --scripted $S --games 1024 2>&1 | grep -E "^[0-9]+v|Error|Traceback"
  done
done
echo DONE $(date +%H:%M:%S)
