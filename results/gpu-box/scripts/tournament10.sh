#!/bin/bash
# Tournament 10 (the author, 15:40 KST): the final one, for the Elo plot of all three agent types.
# Adds the nearest model's whole BC -> RL curve (uniform every ~20M, live at BC 30/60/120M and
# RL 60.8/121.6/180.4M, its RL 180.4M exponential) and the Yemong RL at 141.6M and 160.4M
# (uniform) to tournament 7's field, reusing tournaments 5-9's games. scripted = 1000;
# 8 x 256, target +-35.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }
until grep -q "done 2026-11-10-rl-nearest\|FAILED 2026-11-10-rl-nearest" "$LOG"; do sleep 20; done
grep -q "done 2026-11-10-rl-nearest" "$LOG" || { say "tournament10 not started: the nearest RL failed"; exit 1; }
B=runs/2026-11-10-bc-nearest/policies
R=runs/2026-11-10-rl-nearest/policies
players=()
for s in 20971520 30408704 40894464 60817408 80740352 100663296 120586240; do
  players+=($B/step_$s/uniform_average)
done
players+=($B/step_30408704/live $B/step_60817408/live $B/step_120586240/live
  $B/step_120586240/exponential_average)
for s in 20971520 40894464 60817408 80740352 100663296 121634816 141557760 160432128 180355072; do
  [ -d $R/step_$s/uniform_average ] && players+=($R/step_$s/uniform_average)
done
players+=($R/step_60817408/live $R/step_121634816/live $R/step_180355072/live
  $R/step_180355072/exponential_average)
# the Yemong lines' whole curves (tournament 7's points, games reused) and their RL past 121.6M:
# reused games count only between players listed here, so the plot's three lines share a field
for line in deadreckon-split-hiddenphysics absolute; do
  for s in 10485760 20971520 30408704 40894464 50331648 60817408 70254592 80740352 90177536 \
      100663296 110100480 120586240; do
    players+=(runs/2026-11-09-bc-$line/policies/step_$s/uniform_average)
  done
  for s in 10485760 20971520 30408704 40894464 50331648 60817408 70254592 80740352 90177536 \
      100663296 110100480 121634816 141557760 160432128 180355072; do
    players+=(runs/2026-11-09-rl-$line/policies/step_$s/uniform_average)
  done
done
for p in "${players[@]}"; do [ -f $p/weights.safetensors ] || { say "tournament10: missing $p"; exit 1; }; done
say "tournament10 start (${#players[@]} added players: the nearest curve, the Yemong RL to 180M)"
if .venv/bin/bnb ratings 2026-11-09-bc-absolute --no-league --no-ladder --with "${players[@]}" \
    --presets frontline deathmatch --target-stderr 35 --blocks 8 --block-environments 256 \
    --reuse runs/ratings/round5 runs/ratings/round6 runs/ratings/round7-8x256 runs/ratings/round8 \
    runs/ratings/round9 --out runs/ratings/round10 >> runs/ratings-round10.out 2>&1; then
  say "tournament10 done"
else
  say "tournament10 FAILED (runs/ratings-round10.out)"
fi
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
