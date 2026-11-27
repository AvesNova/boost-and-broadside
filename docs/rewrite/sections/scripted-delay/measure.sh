#!/bin/bash
# The 1v1 aim-error and hit-rate measurement, old and new, yardstick and robust centre (CPU).
cd "$(dirname "$0")/../../../.."
D=docs/rewrite/sections/scripted-delay
C=docs/rewrite/sections/lookahead-personas/candidates/robust-centre-3.json
for v in old new; do
  PYTHONPATH=$PWD uv run --no-sync python $D/measure.py $v --envs 128 --ticks 1800 --out $D/aim-$v-yardstick.json > /dev/null
  PYTHONPATH=$PWD uv run --no-sync python $D/measure.py $v --envs 128 --ticks 1800 --rows $C --out $D/aim-$v-centre.json > /dev/null
  echo "$(date +%H:%M:%S) $v done"
done
