#!/bin/bash
# Merge the seed-1 sweeps with the seed-2 extra games, then draw every fan chart.
#   finalize.sh [SIZE ...]    sizes to draw fan charts for (default 4 8 16 32)
# Expects $FAN_DIR/fan_731_n{N}.base.jsonl (the seed-1 sweep, renamed) and
# fan_731_n{N}.extra.jsonl; writes fan_731_n{N}.jsonl and the charts beside them.
set -e
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
F=${FAN_DIR:-$REPO/out/studies/fan}
case $F in /*) ;; *) F=$REPO/$F ;; esac
cd "$F"
SIZES=${*:-4 8 16 32}
declare -A T=([4]="Four trained ships against more" [8]="Eight trained ships against more" [16]="Sixteen trained ships against more" [32]="Thirty-two trained ships against more")
PY="env VIRTUAL_ENV= uv run --no-sync --project $REPO python"
$PY "$HERE/merge_extra.py" $SIZES
for N in $SIZES; do $PY "$HERE/plot_fan.py" fan_731_n$N.jsonl fan_731_n$N "${T[$N]}"; done
$PY "$HERE/crossover.py" crossover.json $(ls fan_731_n{4,8,16,32}.jsonl 2>/dev/null)
$PY "$HERE/extras.py"
$PY "$HERE/win_rate_by_size.py"
