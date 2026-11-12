#!/bin/bash
# round.sh STAGE MODE CONFIG STEPS THREADS VARIANT... : the variants' measurements of one stage
# side by side, each in its own tmux session (measure.sh), THREADS compile threads each.
# Variants: A (today), B (compile_regions), C (compile_split), BC (both).
set -euo pipefail
STAGE=$1; MODE=$2; CONFIG=$3; STEPS=$4; THREADS=$5; shift 5
declare -A SETS=([A]="" [B]="compile_regions=true" [C]="compile_split=true"
                 [BC]="compile_regions=true;compile_split=true")
for variant in "$@"; do
  tmux new -d -s "$variant-$STAGE-$MODE" \
    "TORCHINDUCTOR_COMPILE_THREADS=$THREADS bash /workspace/boost-and-broadside/docs/rewrite/sections/compile-split/measure.sh $variant $STAGE $MODE $CONFIG $STEPS '${SETS[$variant]}' >> /workspace/cs/rounds.out 2>&1"
  sleep 5
done
