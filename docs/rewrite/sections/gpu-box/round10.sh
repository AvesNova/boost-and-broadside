#!/bin/bash
# Round 10 (the author, 04:10 KST): both RL runs to 120M (116 updates = 121.6M), then tournament 7:
# the whole curve of both headline runs in one field so every Elo is directly comparable: each
# run's uniform average every ~10M from the first BC snapshot to the last RL one (BC 10.5 ...
# 120.6M; RL 10.5 ... 121.6M; RL step 0 = the BC 120.6M uniform it started from), plus live vs
# uniform vs exponential at BC 120.6M, RL 60.8M and RL 121.6M, and the scripted agent (1000).
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
until grep -q "round9 done" "$LOG"; do sleep 30; done

for name in 2026-11-09-rl-absolute 2026-11-09-rl-deadreckon-split-hiddenphysics; do
  if [ -d runs/$name/policies/step_121634816 ]; then say "$name already at 121.6M"; continue; fi
  say "start resume $name to 116 updates (121.6M)"
  if .venv/bin/bnb train --resume $name --updates 116 >> runs/$name.out 2>&1; then
    say "done resume $name to 121.6M"
  else
    say "FAILED resume $name (runs/$name.out)"
  fi
  docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
done

BC_STEPS="10485760 20971520 30408704 40894464 50331648 60817408 70254592 80740352 90177536 100663296 110100480 120586240"
RL_STEPS="10485760 20971520 30408704 40894464 50331648 60817408 70254592 80740352 90177536 100663296 110100480 121634816"
players=()
add() { if [ -f "$1/checkpoint.json" ]; then players+=("$1"); else say "tournament7: missing $1"; fi; }
for line in deadreckon-split-hiddenphysics absolute; do
  for s in $BC_STEPS; do add runs/2026-11-09-bc-$line/policies/step_$s/uniform_average; done
  for s in $RL_STEPS; do add runs/2026-11-09-rl-$line/policies/step_$s/uniform_average; done
  for v in live exponential_average; do
    add runs/2026-11-09-bc-$line/policies/step_120586240/$v
    add runs/2026-11-09-rl-$line/policies/step_60817408/$v
    add runs/2026-11-09-rl-$line/policies/step_121634816/$v
  done
done
# the author, 04:30 KST: 8 blocks x 256 games (twice the games per network call; 2048 games, under
# training's rollout) and target +-35 (games scale with 1/error^2); the default layout if it fails
for layout in "8 256" "8 128"; do
  set -- $layout
  say "tournament7 start (${#players[@]} players: the whole BC + RL curve; $1 x $2, target 35)"
  if .venv/bin/bnb ratings 2026-11-09-bc-absolute --no-league --no-ladder --with "${players[@]}" \
      --presets frontline deathmatch --budget 60000 --target-stderr 35 \
      --blocks $1 --block-environments $2 \
      --reuse runs/ratings/round5 runs/ratings/round6 \
      --out runs/ratings/round7-$1x$2 >> runs/ratings-round7.out 2>&1; then
    say "tournament7 done ($1 x $2)"; break
  fi
  say "tournament7 FAILED with $1 x $2 (runs/ratings-round7.out)"
done
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
