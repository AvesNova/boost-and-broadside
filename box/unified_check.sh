#!/bin/bash
# unified_check.sh CHAIN [HOURS]: the 5090 check of the unified evaluator (`bnb unified`,
# docs/guides/unified-evaluation.md) on the dry run's policies, the box queue's targets
# (box/eval_queue.sh's, less the map sizes: evaluation plays the trained map sizes) in one run:
#   suite      the champion against Imugi, every head-to-head condition (score to +-0.02)
#   curves     5v5 Elo of both stages' scheduled snapshots per mode (+-60, to precision)
#   breakeven  the champion's break-even ratio per mode, at 5 ships against Imugi's 1 to 20
#              (+-0.1, at most 2048 games)
#   teamsize   the champion at n v n, n v 1.5n, n v 2n, n = 1..10, per mode (+-0.05)
# Launch detached; it takes the GPU lock:
#   ssh -f box5090 'nohup setsid bash /workspace/wt/ueval/box/unified_check.sh dry-v24-s1 2 \
#     > /workspace/runs/dry-v24-s1-unified.launch.log 2>&1 < /dev/null &'
# CHAIN may be several, comma-separated (one field: every chain's snapshots and champion;
# box/unified_spec.py); OLD_IMUGI=1 adds game v24's Imugi as "imugi_v24" (the anchor stays
# this tree's Imugi). PRIOR may name several pools or unified run directories (space-
# separated). PARTS="curves" (or any of: suite curves breakeven teamsize; default all) picks the
# targets; CURVE_DESIGN=static (default: fixed counts in two stages, the anchor and each
# run's checkpoints, bnb.league.unified.static_target; REACH chain neighbours a side, default 2;
# PILOT stage 1's goal factor, default 1: the full prior design) or adaptive (the per-game
# planner); LINEUP_TYPES the lineups' ship types (default "Yongma Jujak"); PRESETS="MODE=PRESET ..."
# overrides a mode's preset (the release run: elimination=real_scale_elimination_hemisphere);
# WINDOWS=SIZE,OVERLAP fixed windows of the training order for the static curves (e.g. 16,8;
# each played out, then the next loaded in place; default: the sliding resident window);
# RESIDENTS the resident window's network players (default 12); PRIOR=POOL loads a rating pool's games first (`--prior`, e.g. the queue's
# $RUNS/CHAIN-eval/curves/pool); OUT overrides the output directory ($RUNS/CHAIN-unified).
# A compile-cache archive (docs/rewrite/sections/launch/cache.sh export) in $CACHE_ARCHIVE
# is imported first, so a box of the same GPU model starts warm. Writes $RUNS/CHAIN-unified/
# (progress.log with games per hour, GPU utilisation and the ETA as a clock time;
# games.jsonl, report.json, sizing.json); a rerun resumes from games.jsonl.
#
# What to expect (the laptop's measurement scaled, docs/rewrite/sections/unified-eval/
# results.md): the 5090 has 170 SMs at about 2.4 GHz against the laptop 4070's 36 at about
# 2.2 GHz, 5.2x by SMs times clock; the measured training-rollout ratio is 3.8x. Each
# bucket's sizing.json should hold about 4x the laptop's environments (32 GB against 8),
# and games per hour should come out 3.8-5x the laptop's per bucket. The GPU should stay
# above 90% busy once warm (progress.log's GPU column); lower means the host loop is the
# limit (check the python process's CPU in top).
set -uo pipefail
chain=${1:?usage: unified_check.sh CHAIN[,CHAIN...] [HOURS]}
hours=${2:-2}
EVAL_ENV=${EVAL_ENV:-/workspace/env.sh}
[ "$EVAL_ENV" != none ] && source "$EVAL_ENV"
UEVAL=${UEVAL:-/workspace/wt/ueval}
PY=${PY:-/workspace/venv/bin/python}
RUNS=${RUNS:-/workspace/runs}
export PYTHONPATH=$UEVAL/src:$UEVAL
out=${OUT:-$RUNS/${chain%%,*}-unified}
mkdir -p "$out"
if [ -n "${CACHE_ARCHIVE:-}" ] && [ -s "$CACHE_ARCHIVE" ]; then
  bash "$UEVAL/docs/rewrite/sections/launch/cache.sh" import "$CACHE_ARCHIVE"
fi
"$PY" "$UEVAL/box/unified_spec.py" "$RUNS" "$chain" \
  --parts "${PARTS:-suite curves breakeven teamsize}" ${OLD_IMUGI:+--old-imugi} --curve-design "${CURVE_DESIGN:-static}" --reach "${REACH:-2}" --pilot "${PILOT:-1}" ${WINDOWS:+--windows "$WINDOWS"} --modes "${MODES:-frontline deathmatch elimination}" --lineup-types "${LINEUP_TYPES:-Yongma Jujak}" ${PRESETS:+$(for p in $PRESETS; do printf -- "--preset %s " "$p"; done)} \
  > "$out/spec.toml" || exit 1
echo "spec: $out/spec.toml"
exec 9> "${GPU_LOCK:-/tmp/gpu.lock}"
flock 9
"$PY" -c 'import sys; from bnb.cli import main; sys.argv[0] = "bnb"; sys.exit(main())' \
  unified "$out/spec.toml" --out "$out" --runs-directory "$RUNS" --hours "$hours" \
  ${RESIDENTS:+--residents $RESIDENTS} ${PRIOR:+--prior $PRIOR}
code=$?
echo "$code" > "$out/exit"
if [ -n "${CACHE_EXPORT:-}" ]; then
  bash "$UEVAL/docs/rewrite/sections/launch/cache.sh" export "$CACHE_EXPORT"
fi
exit $code
