#!/bin/bash
# render.sh N0 N1 SCENARIO OUTDIR STEM LOG SEED:PICKS ... ; one locked GPU job per seed.
# Replays a seeded batch and films the picked games (hero.py render), checking each one
# against its line in LOG. Paths are relative to the repository root.
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE/../../.."
export VIRTUAL_ENV=
LOCK=${GPU_LOCK:-out/gpu.lock}
C=checkpoints/wild-universe-731/avg_step_000671846400.pt
N0=$1; N1=$2; SC=$3; OUT=$4; STEM=$5; LOG=$6; shift 6
ENVS=${ENVS:-512}
for SP in "$@"; do
  S=${SP%%:*}; P=${SP#*:}
  echo "== seed $S picks $P $(date +%H:%M:%S)"
  flock "$LOCK" systemd-run --user --scope -q -p MemoryMax=10G \
    uv run --no-sync python "$HERE/hero.py" render $C --n0 $N0 --n1 $N1 --scenario $SC --envs $ENVS --seed $S --pick $P --outdir $OUT --stem $STEM --log $LOG --gif 2>&1 | grep -E "^\{|Error|Traceback"
done
echo DONE $(date +%H:%M:%S)
