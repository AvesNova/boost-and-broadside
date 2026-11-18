#!/bin/bash
# run_batch.sh N0 N1 SCENARIO OUT SEEDS... ; one locked GPU job per pair of seeds.
# Plays headless seeded batches of the final averaged checkpoint (hero.py batch) and
# appends one line per game to OUT. Paths are relative to the repository root.
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE/../../.."
export VIRTUAL_ENV=
LOCK=${GPU_LOCK:-out/gpu.lock}
C=checkpoints/wild-universe-731/avg_step_000671846400.pt
N0=$1; N1=$2; SC=$3; OUT=$4; shift 4
ENVS=${ENVS:-512}
while [ $# -gt 0 ]; do
  S=$1; shift; if [ $# -gt 0 ]; then S="$S,$1"; shift; fi
  echo "== seeds $S $(date +%H:%M:%S)"
  flock "$LOCK" systemd-run --user --scope -q -p MemoryMax=10G \
    uv run --no-sync python "$HERE/hero.py" batch $C --n0 $N0 --n1 $N1 --scenario $SC --envs $ENVS --seeds $S --out $OUT 2>&1 | grep -E "blue wins|Error|Traceback"
done
echo DONE $(date +%H:%M:%S)
