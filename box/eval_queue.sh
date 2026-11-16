#!/bin/bash
# eval_queue.sh CHAIN: the dry run's evaluation queue on the 5090 box, run once the chain
# (box/dry_run.sh CHAIN: CHAIN-bc then CHAIN-rl) has finished. Launch it detached; it waits
# for the GPU lock, which the chain holds until it exits, so it may be launched early:
#   ssh -f box5090 'nohup setsid bash /workspace/bnb-evalq/box/eval_queue.sh dry-v24-s1 \
#     > /workspace/runs/dry-v24-s1-eval.launch.log 2>&1 < /dev/null &'
#
# Runs from its own checkout ($EVALQ, branch feat/eval-queue: crossover's fixed lineups, the
# trail stills and the results charts), not the training's pinned $BNB, with the same venv.
# Writes $RUNS/CHAIN-eval/: progress.log (a line per step with its ETA as a clock time),
# logs/<step>.log (each command's own lines, a line per batch with its ETA), done/<step>
# (a finished step, skipped on a rerun), queue.pid, queue.exit (the failed steps' count,
# 0 when all passed). Every command records its batches as it plays and continues from them,
# so a rerun after a stop resumes inside a step too. The order (the author, 2026-11-17):
#   1 average   the averaging sweep over the RL stage's per-update files, rated in one pool
#               (5v5, the three modes), and the champion (box/eval_champion.py's rule)
#   2 suite     the champion against Imugi on every suite condition (bnb evaluate --games)
#   2 maps      the champion against Imugi at 5v5 on 0.5x and 2x maps (a pool per scale)
#   3 curves    the 5v5 Elo curves of both stages' scheduled snapshots (loose: +-60)
#   4 breakeven bnb crossover on 8 snapshots (BC's last; RL at 5, 10, 20, 35, 50, 75M, last)
#   5 teamsize  the champion at n v n, n v 1.5n, n v 2n, n = 1..10, per mode
#   6 media     trail stills (deathmatch 1v1 and 5v5, elimination 5v5 with fog ghosts) and
#               an MP4 against Imugi (5v5: the play shell has no uneven teams)
#   7 charts    the figures from the records (CPU): curves, fan, break-even, team size, health
# The sizes below are environment variables, so a small test run can shrink them.
set -uo pipefail
chain=${1:?usage: eval_queue.sh CHAIN}
EVAL_ENV=${EVAL_ENV:-/workspace/env.sh}
[ "$EVAL_ENV" != none ] && source "$EVAL_ENV"
EVALQ=${EVALQ:-/workspace/bnb-evalq}
PY=${PY:-/workspace/venv/bin/python}
DEVICE=${DEVICE:-cuda}
export PYTHONPATH=$EVALQ/src:$EVALQ
# sizes (the defaults are the real queue's)
AVERAGE_RUN=${AVERAGE_RUN:-$chain-rl}     # the run whose per-update files are averaged
AVERAGE_SPEC=${AVERAGE_SPEC:-$EVALQ/docs/rewrite/sections/averaging/suite.toml}
AVERAGE_STDERR=${AVERAGE_STDERR:-20}      # each average, per mode
AVERAGE_BUDGET=${AVERAGE_BUDGET:-12288}   # games per mode at most
SUITE_GAMES=${SUITE_GAMES:-512}           # games per suite condition
SUITE_LAYOUT=${SUITE_LAYOUT:-}            # e.g. "--head-to-head-blocks 1 ..." for a test
MAP_SCALES=${MAP_SCALES:-0.5 2.0}
MAP_STDERR=${MAP_STDERR:-45}
CURVE_STDERR=${CURVE_STDERR:-60}          # loose first (the author); 45 later if needed
CURVE_BUDGET=${CURVE_BUDGET:-12288}
POOL_LAYOUT=${POOL_LAYOUT:-}              # e.g. "--blocks 1 --block-environments 4"
BREAK_EVEN_MILLIONS=${BREAK_EVEN_MILLIONS:-5 10 20 35 50 75}
BREAK_EVEN_STDERR=${BREAK_EVEN_STDERR:-0.1}
BREAK_EVEN_GAMES=${BREAK_EVEN_GAMES:-2048} # per mode at most
CROSSOVER_LAYOUT=${CROSSOVER_LAYOUT:---blocks 8 --block-environments 32}
TEAM_SIZES=${TEAM_SIZES:-1 2 3 4 5 6 7 8 9 10}
TEAM_GAMES=${TEAM_GAMES:-128}             # games per lineup
MATCH_SECONDS=${MATCH_SECONDS:-}          # trail games' length (empty: the game's end)
VIDEO_SECONDS=${VIDEO_SECONDS:-40}
FIVES=${FIVES:-frontline_5v5 deathmatch_5v5 elimination_5v5}

out=$RUNS/$chain-eval
mkdir -p "$out/logs" "$out/done"
progress=$out/progress.log
echo $$ > "$out/queue.pid"
say() { echo "$(date '+%F %H:%M:%S') $*" | tee -a "$progress"; }
clock() { date -d "+$1 minutes" '+%H:%M'; }
bnb() { "$PY" -c 'import sys; from bnb.cli import main; sys.argv[0] = "bnb"; sys.exit(main())' "$@"; }
failed=0

# ---- the GPU lock (held by the chain until it exits), then the chain's result
exec 9> "$GPU_LOCK"
if ! flock -n 9; then
  say "waiting for the GPU lock $GPU_LOCK (the chain holds it until it exits)"
  flock 9
fi
say "queue $chain: commit $(git -C "$EVALQ" rev-parse --short=8 HEAD), device $DEVICE"
if [ "${EVAL_FORCE:-0}" != 1 ]; then
  code=$(cat "$RUNS/$chain.exit" 2>/dev/null || echo missing)
  if [ "$code" != 0 ]; then
    say "the chain's exit is $code ($RUNS/$chain.exit), not 0: not evaluating (EVAL_FORCE=1 to)"
    echo 99 > "$out/queue.exit"
    exit 99
  fi
fi
for run in "$chain-bc" "$chain-rl"; do
  [ -d "$RUNS/$run/policies" ] || { say "no $RUNS/$run/policies"; echo 98 > "$out/queue.exit"; exit 98; }
done

# step NAME MINUTES: run step_NAME unless done, with its log and ETA; a failure is counted
step() {
  local name=$1 minutes=$2
  if [ -e "$out/done/$name" ]; then
    say "$name: done before, skipped"
    return
  fi
  say "$name: started, ETA $(clock "$minutes") (about $minutes min on the 5090)"
  local started=$SECONDS
  if "step_$name" >> "$out/logs/$name.log" 2>&1; then
    touch "$out/done/$name"
    say "$name: done in $(((SECONDS - started) / 60)) min"
  else
    failed=$((failed + 1))
    say "$name: FAILED after $(((SECONDS - started) / 60)) min (logs/$name.log)"
  fi
}

champion() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["path"])' "$out/champion.json"; }
need_champion() {
  [ -s "$out/champion.json" ] || { echo "no champion (step average)"; return 1; }
  CHAMPION=$(champion)
  echo "champion $CHAMPION"
}
# the first policy step of RUN at or after MILLIONS (empty: the last)
policy_step() {
  local run=$1 millions=${2:-}
  local steps
  steps=$(ls "$RUNS/$run/policies" | sed -n 's/^step_\([0-9]*\)$/\1/p' | sort -n | awk '$1 > 0')
  if [ -n "$millions" ]; then
    echo "$steps" | awk -v t=$((millions * 1000000)) '$1 >= t {print; exit}'
  else
    echo "$steps" | tail -1
  fi
}

step_average() {
  local dir=$out/averaging
  mkdir -p "$dir"
  local code=0
  bnb average "$AVERAGE_RUN" --spec "$AVERAGE_SPEC" --runs-directory "$RUNS" > "$dir/players.txt" || code=$?
  [ -s "$dir/players.txt" ] || return 1
  [ "$code" = 0 ] || echo "WARNING: bnb average exited $code (a rebuilt training average differs?)"
  bnb pool create "$dir/pool" --lineup-types Yongma Jujak || return 1
  local names=() player added
  while read -r player; do
    if [[ $player == */policies/* ]]; then
      added=$(bnb pool add "$dir/pool" "$RUNS/$player") || return 1
    else
      added=$(bnb pool add "$dir/pool" "$RUNS/$player" --name "$player") || return 1
    fi
    echo "$added"
    names+=("${added%% *}")
  done < "$dir/players.txt"
  bnb pool rate "$dir/pool" --presets $FIVES --target-stderr 45 \
    --decision "${names[@]}" --decision-stderr "$AVERAGE_STDERR" --budget "$AVERAGE_BUDGET" \
    --session-players 16 $POOL_LAYOUT --device "$DEVICE" --runs-directory "$RUNS" || return 1
  bnb pool table "$dir/pool" --presets $FIVES > "$dir/table.txt" || return 1
  "$PY" "$EVALQ/box/eval_champion.py" "$dir/pool" "$out/champion.json" --presets $FIVES \
    | tee "$dir/champion.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || return 1
  say "average: $(head -1 "$dir/champion.txt")"
}

step_suite() {
  need_champion || return 1
  bnb evaluate "$CHAMPION" imugi --out "$out/suite" --games "$SUITE_GAMES" $SUITE_LAYOUT \
    --device "$DEVICE" --runs-directory "$RUNS" || return 1
  bnb evaluate decide "$out/suite" --peek > "$out/suite/summary.txt"
}

step_maps() {
  need_champion || return 1
  local scale pool
  for scale in $MAP_SCALES; do
    pool=$out/maps/scale_$scale
    bnb pool create "$pool" --lineup-types Yongma Jujak --arena-scale "$scale" || return 1
    bnb pool add "$pool" "$CHAMPION" --name champion || return 1
    bnb pool rate "$pool" --presets $FIVES --target-stderr "$MAP_STDERR" --budget 4096 \
      --session-players 2 $POOL_LAYOUT --device "$DEVICE" --runs-directory "$RUNS" || return 1
    bnb pool table "$pool" --presets $FIVES > "$pool/table.txt" || return 1
  done
}

step_curves() {
  local pool=$out/curves/pool
  bnb pool create "$pool" --lineup-types Yongma Jujak || return 1
  bnb pool add "$pool" --runs "$chain-bc" "$chain-rl" --runs-directory "$RUNS" || return 1
  bnb pool rate "$pool" --presets $FIVES --target-stderr "$CURVE_STDERR" \
    --budget "$CURVE_BUDGET" --session-players 64 $POOL_LAYOUT --device "$DEVICE" \
    --runs-directory "$RUNS" || return 1
  bnb pool table "$pool" --presets $FIVES > "$out/curves/table.txt" || return 1
  bnb pool curves "$pool" --presets $FIVES --out "$out/curves/curves.csv"
}

step_breakeven() {
  local run step points=() millions
  points+=("$chain-bc:$(policy_step "$chain-bc")")
  for millions in $BREAK_EVEN_MILLIONS; do
    step=$(policy_step "$chain-rl" "$millions")
    [ -n "$step" ] && points+=("$chain-rl:$step")
  done
  points+=("$chain-rl:$(policy_step "$chain-rl")")
  local point code=0
  for point in $(printf '%s\n' "${points[@]}" | awk '!seen[$0]++'); do
    run=${point%%:*} step=${point##*:}
    local dir=$out/break-even/$run/step_$step
    if ls "$dir"/artifacts/crossover/*/result.json > /dev/null 2>&1; then
      echo "$run step $step: measured before"
      continue
    fi
    say "breakeven: $run step $step"
    bnb crossover "$RUNS/$run/policies/step_$step/exponential_average" \
      --target-stderr "$BREAK_EVEN_STDERR" --budget-games "$BREAK_EVEN_GAMES" \
      $CROSSOVER_LAYOUT --device "$DEVICE" --out "$dir" --runs-directory "$RUNS" || code=1
  done
  return $code
}

step_teamsize() {
  need_champion || return 1
  local n lineups=()
  for n in $TEAM_SIZES; do
    lineups+=("${n}v${n}")
    [ $((n % 2)) = 0 ] && lineups+=("${n}v$((3 * n / 2))")
    lineups+=("${n}v$((2 * n))")
  done
  echo "lineups: ${lineups[*]}"
  bnb crossover "$CHAMPION" --lineups "${lineups[@]}" --lineup-games "$TEAM_GAMES" \
    $CROSSOVER_LAYOUT --device "$DEVICE" --out "$out/team-size" --runs-directory "$RUNS"
}

step_media() {
  need_champion || return 1
  local seconds=() code=0
  [ -n "$MATCH_SECONDS" ] && seconds=(--match.seconds "$MATCH_SECONDS")
  local name preset size
  for name in deathmatch_1v1:real_scale_deathmatch:1 deathmatch_5v5:real_scale_deathmatch:5 \
    elimination_5v5:real_scale_elimination:5; do
    IFS=: read -r name preset size <<< "$name"
    [ -e "$out/media/trails/$name/trails_game.png" ] && continue
    say "media: trails $name"
    bnb figures trails --policy "$CHAMPION" --preset "$preset" --match.preset "$preset" \
      --match.ships-per-team "$size" "${seconds[@]}" --out "$out/media/trails/$name" \
      --runs-directory "$RUNS" || code=1
  done
  if [ ! -s "$out/media/champion_deathmatch_5v5.mp4" ]; then
    say "media: video"
    bnb figures video --policy "$CHAMPION" --preset real_scale_deathmatch \
      --out "$out/media/champion_deathmatch_5v5.mp4" --video.seconds "$VIDEO_SECONDS" \
      --runs-directory "$RUNS" || code=1
  fi
  return $code
}

step_charts() {
  local charts=$out/charts code=0
  mkdir -p "$charts"
  if [ -d "$out/curves/pool" ]; then
    bnb pool curves "$out/curves/pool" --presets $FIVES --out "$out/curves/curves.csv" \
      --figure "$charts/curves.png" --break-even "$out/break-even" || code=1
  fi
  if [ -d "$out/break-even" ]; then
    for chart in fan break-even; do
      bnb figures results $chart "$out/break-even" --out "$charts" \
        --values "$charts/values.json" || code=1
    done
  fi
  if [ -d "$out/team-size" ]; then
    bnb figures results team-size "$out/team-size" --out "$charts" \
      --values "$charts/values.json" || code=1
  fi
  bnb figures results health "$RUNS/$chain-bc" "$RUNS/$chain-rl" --out "$charts" \
    --values "$charts/values.json" || code=1
  return $code
}

step average 45
step suite 35
step maps 15
step curves 55
step breakeven 45
step teamsize 25
step media 25
step charts 3
say "queue $chain: finished, $failed steps failed"
echo "$failed" > "$out/queue.exit"
exit "$failed"
