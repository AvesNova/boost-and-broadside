#!/bin/bash
# hero_clip.sh launch|box|fetch CHAIN: the README's hero clip of the dry run's champion,
# made on the 5090 box after box/eval_queue.sh CHAIN has finished (bnb figures hero:
# docs/guides/figures.md#the-hero-clip).
#   launch (laptop): start the box part detached; its log is $RUNS/CHAIN-eval/media/hero.log
#     bash box/hero_clip.sh launch dry-v24-s1
#   box (on the box): wait for the GPU lock (the queue holds it until it exits: never runs
#     beside it), check out origin/feat/hero-clip at /workspace/wt/hero, film HERO_GAMES
#     games (8) of the champion ($RUNS/CHAIN-eval/champion.json) against Imugi in real-scale
#     elimination at the lineup, pick the clip and make its GIF and poster frame, into
#     $RUNS/CHAIN-eval/media/hero/ (each game's MP4 and JSON as it ends; a rerun continues)
#   fetch (laptop): copy hero.gif, hero.png, chosen.json and the games' JSONs to DEST
#     (default /home/vizia/avesnova/boost-and-broadside-v2/review/writeup/media/)
# The lineup: HERO_SHIPS ("5 7": the champion's ships, then Imugi's), else the elimination
# break-even ratio of the RL stage's last snapshot (the queue's breakeven step) at
# HERO_POLICY_SHIPS (5) of the champion's ships, Imugi's rounded. On the laptop's CPU
# (software OpenGL) a game took 1.4-2.5 s per game second (31-216 s games: 77-520 s); the
# box draws on its GPU, so 8 games of at most 248 s should take well under an hour.
set -uo pipefail
mode=${1:?usage: hero_clip.sh launch|box|fetch CHAIN [DEST]}
chain=${2:?usage: hero_clip.sh launch|box|fetch CHAIN [DEST]}
box=${BOX:-box5090}

if [ "$mode" = launch ]; then
  ssh -f "$box" "mkdir -p /workspace/runs/$chain-eval/media && cd /workspace && \
    git -C /workspace/bnb fetch -q origin feat/hero-clip && \
    git -C /workspace/bnb show origin/feat/hero-clip:box/hero_clip.sh > /workspace/hero_clip.sh && \
    nohup setsid bash /workspace/hero_clip.sh box $chain \
      > /workspace/runs/$chain-eval/media/hero.log 2>&1 < /dev/null &"
  echo "launched; log: $box:/workspace/runs/$chain-eval/media/hero.log"
  exit 0
fi

if [ "$mode" = fetch ]; then
  dest=${3:-/home/vizia/avesnova/boost-and-broadside-v2/review/writeup/media}
  mkdir -p "$dest/hero-games"
  source=/workspace/runs/$chain-eval/media/hero
  rsync -a "$box:$source/hero.gif" "$box:$source/hero.png" "$box:$source/chosen.json" "$dest/" \
    || { echo "no clip yet: $box:/workspace/runs/$chain-eval/media/hero.log"; exit 1; }
  rsync -a --include 'game_seed*.json' --exclude '*' "$box:$source/" "$dest/hero-games/"
  python3 -c 'import json, sys; c = json.load(open(sys.argv[1])); print(c["reason"]); print(c.get("gif"))' \
    "$dest/chosen.json"
  ls -la "$dest/hero.gif" "$dest/hero.png"
  exit 0
fi

[ "$mode" = box ] || { echo "mode $mode: launch, box or fetch"; exit 2; }
source /workspace/env.sh
HERO=${HERO:-/workspace/wt/hero}
PY=${PY:-/workspace/venv/bin/python}
out=$RUNS/$chain-eval
say() { echo "$(date '+%F %H:%M:%S') hero: $*" | tee -a "$out/progress.log"; }
exec 9> "$GPU_LOCK"
if ! flock -n 9; then
  say "waiting for the GPU lock $GPU_LOCK (the evaluation queue holds it until it exits)"
  flock 9
fi
[ -s "$out/champion.json" ] || { say "no $out/champion.json (the queue's average step)"; exit 1; }
champion=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["path"])' "$out/champion.json")
git -C "$BNB" fetch -q origin feat/hero-clip || exit 1
if [ -d "$HERO" ]; then
  git -C "$HERO" checkout -q --detach origin/feat/hero-clip || exit 1
else
  git -C "$BNB" worktree add -q --detach "$HERO" origin/feat/hero-clip || exit 1
fi
export PYTHONPATH=$HERO/src:$HERO

if [ -n "${HERO_SHIPS:-}" ]; then
  ships=$HERO_SHIPS
else
  step=$(ls "$out/break-even/$chain-rl" 2>/dev/null | sed -n 's/^step_\([0-9]*\)$/\1/p' | sort -n | tail -1)
  result=$(ls "$out/break-even/$chain-rl/step_$step"/artifacts/crossover/*/result.json 2>/dev/null | tail -1)
  [ -n "$result" ] || { say "no break-even result under $out/break-even (HERO_SHIPS to set one)"; exit 1; }
  ships=$("$PY" - "$result" "${HERO_POLICY_SHIPS:-5}" <<'EOF'
import json, sys
presets = json.load(open(sys.argv[1]))["payload"]["presets"]
name = next(name for name in presets if "elimination" in name)
ratio, own = presets[name]["ratio"], int(sys.argv[2])
if ratio is None:
    sys.exit(f"{name}: no break-even ratio ({presets[name]['fit']['status']})")
print(own, max(1, round(own * ratio)))
print(f"{name}: break-even ratio {ratio:.3f} (lineup {presets[name]['lineup']})", file=sys.stderr)
EOF
) || { say "no lineup from $result"; exit 1; }
fi
say "champion $champion at ${ships/ /v} in real-scale elimination, ${HERO_GAMES:-8} games, commit $(git -C "$HERO" rev-parse --short=8 HEAD)"
# shellcheck disable=SC2086
"$PY" -c 'import sys; from bnb.cli import main; sys.argv[0] = "bnb"; sys.exit(main())' \
  figures hero --policy "$champion" --preset real_scale_elimination --hero.ships $ships \
  --hero.games "${HERO_GAMES:-8}" --out "$out/media/hero" --gif "$out/media/hero/hero.gif" \
  --device "${DEVICE:-cuda}" --runs-directory "$RUNS"
code=$?
say "finished (exit $code): $(head -c 300 "$out/media/hero/chosen.json" 2>/dev/null | tr '\n' ' ')"
exit $code
