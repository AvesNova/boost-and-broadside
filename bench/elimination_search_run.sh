#!/bin/bash
# Imugi's search and edge steer, and the shrink laws (2026-11-15, CPU, eager): every
# run 5v5, fields on, mirrored lineups, seed 61, one setting at a time. Usage:
#   bench/elimination_search_run.sh OUT_DIR [PHASE...]   (phases: search edge laws deathmatch)
set -u
cd "$(dirname "$0")/.."
OUT=${1:?out dir}; shift
PHASES=${*:-search edge laws deathmatch}
mkdir -p "$OUT"
GAMES=${GAMES:-128}
run() {  # label kind out-file arguments...
  local label=$1 kind=$2 file=$3; shift 3
  echo "== $label start $(date +%H:%M:%S)"
  uv run --no-sync python -m bench.mode_dynamics "$kind" --games "$GAMES" --batch "$GAMES" \
    --fields --eager --device cpu --lineup "yongma,jujak;jujak,yongma" --seed 61 \
    --label "$label" --out "$OUT/$file" "$@" > "$OUT/$label.log" 2>&1
  echo "== $label done $(date +%H:%M:%S) rc=$?"
}
for phase in $PHASES; do case $phase in
search)  # no shrink (the hold outlasts the clock), 7.8 km, 300 s
  for search in centre straight explore; do
    run "search-$search" elimination search.jsonl --episode-seconds 300 \
      --set mode.shrink_hold_seconds=100000 --imugi-search "$search"
  done ;;
edge)  # the fastest linear shrink (start 0, 60 s), still against moving edge steer
  run edge-still elimination edge.jsonl --episode-seconds 68 --set mode.shrink_hold_seconds=0 \
    --imugi-search explore --imugi-still-edge
  run edge-moving elimination edge.jsonl --episode-seconds 68 --set mode.shrink_hold_seconds=0 \
    --imugi-search explore ;;
laws)  # start 0, cap 150 m/s, moving edge steer, explore search; clock = shrink + 8 s
  for T in 60 90 120 180; do
    run "linear-$T" elimination laws.jsonl --episode-seconds $((T + 8)) \
      --set mode.shrink_hold_seconds=0 --imugi-search explore
  done
  for p in 2 3; do
    run "power$p-120" elimination laws.jsonl --episode-seconds 128 --set mode.shrink_hold_seconds=0 \
      --set mode.shrink_law=2 --set mode.shrink_power=$p --imugi-search explore
  done
  for spec in 0.32:120 0.32:180 0.256:180; do
    k=${spec%%:*}; T=${spec##*:}
    run "corner$k-$T" elimination laws.jsonl --episode-seconds $((T + 8)) \
      --set mode.shrink_hold_seconds=0 --set mode.shrink_law=3 --set mode.shrink_knee_seconds=30 \
      --set mode.shrink_knee_fraction=$k --imugi-search explore
  done ;;
interaction)  # the search under a mid shrink (linear 120 s; explore's is in laws)
  for search in centre straight; do
    run "linear-120-$search" elimination laws.jsonl --episode-seconds 128 \
      --set mode.shrink_hold_seconds=0 --imugi-search "$search"
  done ;;
nocap)  # the fast-then-slow laws without the speed cap (how much the cap held them back)
  run power3-120-nocap elimination laws.jsonl --episode-seconds 128 --set mode.shrink_hold_seconds=0 \
    --set mode.shrink_law=2 --set mode.shrink_power=3 --set mode.shrink_max_speed=0 \
    --imugi-search explore
  run corner0.32-120-nocap elimination laws.jsonl --episode-seconds 128 \
    --set mode.shrink_hold_seconds=0 --set mode.shrink_law=3 --set mode.shrink_knee_seconds=30 \
    --set mode.shrink_knee_fraction=0.32 --set mode.shrink_max_speed=0 --imugi-search explore ;;
sizes)  # other team sizes on the fixed maps: elimination with linear 120 s, deathmatch
  for n in 1 2 10; do
    run "size-elim-${n}v${n}" elimination sizes.jsonl --ships $n --episode-seconds 128 \
      --set mode.shrink_hold_seconds=0 --imugi-search explore
  done
  for n in 1 2; do
    GAMES=64 run "size-dm-${n}v${n}" deathmatch sizes.jsonl --ships $n --imugi-search explore
  done
  GAMES=32 run size-dm-10v10 deathmatch sizes.jsonl --ships 10 --imugi-search explore ;;
explore)  # the explore search (tight and wide spacing) against the old centre rule, with
  # the independent spawn centres: no shrink, linear 120 s, and the deathmatch kill rate
  run x-none-centre elimination explore.jsonl --episode-seconds 300 \
    --set mode.shrink_hold_seconds=100000 --imugi-search centre
  for spacing in tight wide; do
    run "x-none-$spacing" elimination explore.jsonl --episode-seconds 300 \
      --set mode.shrink_hold_seconds=100000 --imugi-search explore --imugi-spacing $spacing
  done
  run x-lin120-centre elimination explore.jsonl --episode-seconds 128 \
    --set mode.shrink_hold_seconds=0 --imugi-search centre
  for spacing in tight wide; do
    run "x-lin120-$spacing" elimination explore.jsonl --episode-seconds 128 \
      --set mode.shrink_hold_seconds=0 --imugi-search explore --imugi-spacing $spacing
  done
  for spacing in tight wide; do
    GAMES=64 run "x-dm-$spacing" deathmatch explore.jsonl --imugi-search explore \
      --imugi-spacing $spacing
  done ;;
frozen)  # the frozen game (version 21, the presets as they are) with today's Imugi
  run v21-elimination elimination frozen.jsonl
  run v21-frontline frontline frozen.jsonl
  run v21-deathmatch deathmatch frozen.jsonl ;;
deathmatch)  # the kill rate by search (tickets off, the preset's 600 s)
  for search in centre straight explore; do
    GAMES=64 run "dm-$search" deathmatch deathmatch.jsonl --imugi-search "$search"
  done ;;
esac; done
