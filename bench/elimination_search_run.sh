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
  for search in centre straight waypoint; do
    run "search-$search" elimination search.jsonl --episode-seconds 300 \
      --set mode.shrink_hold_seconds=100000 --imugi-search "$search"
  done ;;
edge)  # the fastest linear shrink (start 0, 60 s), still against moving edge steer
  run edge-still elimination edge.jsonl --episode-seconds 68 --set mode.shrink_hold_seconds=0 \
    --imugi-search waypoint --imugi-still-edge
  run edge-moving elimination edge.jsonl --episode-seconds 68 --set mode.shrink_hold_seconds=0 \
    --imugi-search waypoint ;;
laws)  # start 0, cap 150 m/s, moving edge steer, waypoint search; clock = shrink + 8 s
  for T in 60 90 120 180; do
    run "linear-$T" elimination laws.jsonl --episode-seconds $((T + 8)) \
      --set mode.shrink_hold_seconds=0 --imugi-search waypoint
  done
  for p in 2 3; do
    run "power$p-120" elimination laws.jsonl --episode-seconds 128 --set mode.shrink_hold_seconds=0 \
      --set mode.shrink_law=2 --set mode.shrink_power=$p --imugi-search waypoint
  done
  for spec in 0.32:120 0.32:180 0.256:180; do
    k=${spec%%:*}; T=${spec##*:}
    run "corner$k-$T" elimination laws.jsonl --episode-seconds $((T + 8)) \
      --set mode.shrink_hold_seconds=0 --set mode.shrink_law=3 --set mode.shrink_knee_seconds=30 \
      --set mode.shrink_knee_fraction=$k --imugi-search waypoint
  done ;;
interaction)  # the search under a mid shrink (linear 120 s; waypoint's is in laws)
  for search in centre straight; do
    run "linear-120-$search" elimination laws.jsonl --episode-seconds 128 \
      --set mode.shrink_hold_seconds=0 --imugi-search "$search"
  done ;;
nocap)  # the fast-then-slow laws without the speed cap (how much the cap held them back)
  run power3-120-nocap elimination laws.jsonl --episode-seconds 128 --set mode.shrink_hold_seconds=0 \
    --set mode.shrink_law=2 --set mode.shrink_power=3 --set mode.shrink_max_speed=0 \
    --imugi-search waypoint
  run corner0.32-120-nocap elimination laws.jsonl --episode-seconds 128 \
    --set mode.shrink_hold_seconds=0 --set mode.shrink_law=3 --set mode.shrink_knee_seconds=30 \
    --set mode.shrink_knee_fraction=0.32 --set mode.shrink_max_speed=0 --imugi-search waypoint ;;
deathmatch)  # the kill rate by search (tickets off, the preset's 600 s)
  for search in centre straight waypoint; do
    GAMES=64 run "dm-$search" deathmatch deathmatch.jsonl --imugi-search "$search"
  done ;;
esac; done
