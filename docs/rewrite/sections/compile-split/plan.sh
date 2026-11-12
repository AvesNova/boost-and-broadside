#!/bin/bash
# plan.sh: the rest of the measurement as a chain on the box, at most three processes on
# the 8 GB GPU (four ran out of memory in the first learn). Logs to /workspace/cs/plan.log.
D=/workspace/boost-and-broadside/docs/rewrite/sections/compile-split
E=/workspace/cs/configs/e200
say() { echo "$(TZ=Asia/Seoul date +%H:%M:%S) $*" >> /workspace/cs/plan.log; }
wait_for() { while tmux ls 2>/dev/null | grep -qE "^($(echo "$@" | tr ' ' '|')):"; do sleep 20; done; }
say "plan start"
wait_for A-bc-cold B-bc-cold
say "round 1 A, B done; BC bc cold and the trace-only ablations"
bash $D/round.sh bc cold $E/bc.json 1 3 BC
tmux new -d -s ablate-eager "bash $D/ablate.sh eager bc-full bc-nolanes bc-frontline bc-frontline-nolanes ppo-full ppo-nolanes ppo-nolanes-slot1 ppo-frontline ppo-frontline-nolanes-slot1 >> /workspace/cs/rounds.out 2>&1"
wait_for C-bc-cold
say "C bc done; PPO A, C cold"
bash $D/round.sh ppo cold $E/ppo.json 1 3 A C
wait_for BC-bc-cold ablate-eager
say "BC bc and the ablations done; PPO B, BC cold wait for A, C's learn"
wait_for A-ppo-cold C-ppo-cold
say "PPO A, C done; PPO B, BC cold, A width 192 bc cold"
bash $D/round.sh ppo cold $E/ppo.json 1 3 B BC
tmux new -d -s A192-bc-cold "TORCHINDUCTOR_COMPILE_THREADS=3 bash $D/measure.sh A192 bc cold /workspace/cs/configs/e200w192/bc.json 1 '' >> /workspace/cs/rounds.out 2>&1"
wait_for B-ppo-cold BC-ppo-cold A192-bc-cold
say "cold rounds done; warm runs alone"
declare -A SETS=([A]="" [B]="compile_regions=true" [C]="compile_split=true" [BC]="compile_regions=true;compile_split=true")
for stage in bc ppo; do
  for variant in A C B BC; do
    say "warm $variant $stage"
    TORCHINDUCTOR_COMPILE_THREADS=10 bash $D/measure.sh $variant $stage warm $E/$stage.json 4 "${SETS[$variant]}" >> /workspace/cs/rounds.out 2>&1
  done
done
say "plan done"
