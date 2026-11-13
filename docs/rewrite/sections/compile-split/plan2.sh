#!/bin/bash
# plan2.sh: after round 1 (re-plan 19:45): the tree with the heads' sums eager and equal
# micro-batches, micro-batches of 250 rows (25 views: one shape, as the 5090's BC), at most
# two compiling processes (each holds about 12 GB of host memory in the learner's compile;
# three hit the box's 32 GB and the OOM killer).
D=/workspace/boost-and-broadside/docs/rewrite/sections/compile-split
N=/workspace/cs/configs/new
say() { echo "$(TZ=Asia/Seoul date +%H:%M:%S) $*" >> /workspace/cs/plan.log; }
wait_for() { while tmux ls 2>/dev/null | grep -qE "^($(echo "$@" | tr ' ' '|')):"; do sleep 20; done; }
run() {  # name variant stage mode config steps sets threads
  tmux new -d -s "$1" "TORCHINDUCTOR_COMPILE_THREADS=$8 bash $D/measure.sh $2 $3 $4 $5 $6 '$7' >> /workspace/cs/rounds.out 2>&1"
}
say "plan2 start: waiting for round 1's A and BC"
wait_for A-bc-cold BC-bc-cold
cd /workspace/boost-and-broadside && git pull -q
say "pulled $(git log --oneline -1 | cut -c1-70)"
bash $D/configs.sh $N 200 learner.micro_batch_rows=250
bash $D/configs.sh $N-w192 200 learner.micro_batch_rows=250 yemong.width=192
say "new A and C bc cold, side by side"
run newA-bc-cold newA bc cold $N/bc.json 1 "" 5
run newC-bc-cold newC bc cold $N/bc.json 1 "compile_split=true" 5
wait_for newA-bc-cold newC-bc-cold
for variant in newA newC; do
  sets=""; [ $variant = newC ] && sets="compile_split=true"
  say "warm $variant bc alone"
  TORCHINDUCTOR_COMPILE_THREADS=10 bash $D/measure.sh $variant bc warm $N/bc.json 4 "$sets" >> /workspace/cs/rounds.out 2>&1
done
say "new A ppo cold and A width 192 bc cold, side by side"
run newA-ppo-cold newA ppo cold $N/ppo.json 1 "" 5
run newA192-bc-cold newA192 bc cold $N-w192/bc.json 1 "" 5
wait_for newA-ppo-cold newA192-bc-cold
say "warm new A ppo alone"
TORCHINDUCTOR_COMPILE_THREADS=10 bash $D/measure.sh newA ppo warm $N/ppo.json 4 "" >> /workspace/cs/rounds.out 2>&1
say "plan2 done"
